#!/usr/bin/env python3
"""NFL difference engine.

Every completed regular-season and playoff game since the 1970 merger is one
score difference: home rating minus away rating, plus a venue edge when the
site is not neutral. Solving those differences rates each team. The next game
that has not kicked off goes to the side the ratings favor.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_PATH = Path(os.environ.get("NFL_HISTORY_CACHE", str(ROOT / "data" / "nfl_history.json")))

FIRST_SEASON = 1970
HALF_LIFE_SEASONS = 4.0
TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SCHEDULE_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/schedule"
    "?season={season}&seasontype={season_type}"
)

METHOD = (
    "Each finished NFL game since 1970 is a score difference: the home team's "
    "rating minus the away team's rating, plus the venue edge when the field is "
    "not neutral. A game from four seasons ago counts half as much as one from "
    "this season. The engine solves those differences for every team, then "
    "applies the same sum to the next game that has not kicked off."
)

_lock = threading.Lock()
_memory: dict = {"report": None, "games": None, "teams": None}


def fetch_json(url: str, attempts: int = 4):
    """GET JSON. A browser-like user agent is rejected by this feed, so leave it default."""
    last = None
    for i in range(attempts):
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 404:
                return None
            if exc.code in (429, 500, 502, 503, 504):
                time.sleep(0.4 * (2**i))
                continue
            raise
        except urllib.error.URLError as exc:
            last = exc
            time.sleep(0.4 * (2**i))
    if last:
        raise last
    return None


def _score_value(score):
    if score is None:
        return None
    if isinstance(score, dict):
        raw = score.get("value")
        if raw is None:
            raw = score.get("displayValue")
    else:
        raw = score
    if raw is None or raw == "":
        return None
    return int(round(float(raw)))


def _identity(competitor: dict) -> dict:
    team = competitor.get("team") or {}
    return {
        "id": str(team.get("id") or ""),
        "name": team.get("displayName") or team.get("name") or "Unknown",
        "abbreviation": team.get("abbreviation") or "",
    }


def games_from_schedule(payload: dict | None) -> list[dict]:
    """Completed games from one team schedule payload."""
    if not payload:
        return []
    games = []
    for event in payload.get("events") or []:
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        comp = competitions[0]
        status = ((comp.get("status") or {}).get("type") or {})
        if not status.get("completed") and status.get("state") != "post":
            continue
        sides = {}
        for competitor in comp.get("competitors") or []:
            score = _score_value(competitor.get("score"))
            if score is None:
                continue
            place = competitor.get("homeAway")
            if place not in ("home", "away"):
                continue
            ident = _identity(competitor)
            if not ident["id"]:
                continue
            ident["score"] = score
            sides[place] = ident
        if "home" not in sides or "away" not in sides:
            continue
        year = (event.get("season") or {}).get("year")
        if year is None:
            continue
        season_type = event.get("seasonType") or {}
        if isinstance(season_type, dict):
            type_id = int(season_type.get("type") or season_type.get("id") or 2)
        else:
            type_id = int(season_type or 2)
        week = event.get("week") or {}
        week_no = week.get("number") if isinstance(week, dict) else week
        games.append(
            {
                "id": str(event.get("id")),
                "season": int(year),
                "season_type": type_id,
                "week": int(week_no) if week_no is not None else None,
                "date": event.get("date") or "",
                "neutral": bool(comp.get("neutralSite")),
                "home_id": sides["home"]["id"],
                "away_id": sides["away"]["id"],
                "home_name": sides["home"]["name"],
                "away_name": sides["away"]["name"],
                "home_abbr": sides["home"]["abbreviation"],
                "away_abbr": sides["away"]["abbreviation"],
                "home_score": sides["home"]["score"],
                "away_score": sides["away"]["score"],
            }
        )
    return games


def merge_games(*groups: list[dict]) -> list[dict]:
    by_id = {}
    for group in groups:
        for game in group:
            by_id[game["id"]] = game
    return list(by_id.values())


def parse_teams(payload: dict) -> dict:
    leagues = payload["sports"][0]["leagues"][0]
    teams = {}
    for row in leagues.get("teams") or []:
        team = row.get("team") or row
        logos = team.get("logos") or []
        logo = ""
        for item in logos:
            if "default" in (item.get("rel") or []):
                logo = item.get("href") or ""
                break
        if not logo and logos:
            logo = logos[0].get("href") or ""
        team_id = str(team.get("id"))
        teams[team_id] = {
            "id": team_id,
            "name": team.get("displayName") or team.get("name") or team_id,
            "abbreviation": team.get("abbreviation") or "",
            "nickname": team.get("nickname") or team.get("name") or "",
            "color": (team.get("color") or "c4a35a").lstrip("#"),
            "logo": logo,
        }
    return teams


def parse_scoreboard(payload: dict) -> dict:
    upcoming = []
    in_progress = 0
    for event in payload.get("events") or []:
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        comp = competitions[0]
        state = ((comp.get("status") or {}).get("type") or {}).get("state")
        if state == "in":
            in_progress += 1
            continue
        if state != "pre":
            continue
        home = away = None
        for competitor in comp.get("competitors") or []:
            ident = _identity(competitor)
            if not ident["id"]:
                continue
            if competitor.get("homeAway") == "home":
                home = ident
            elif competitor.get("homeAway") == "away":
                away = ident
        if not home or not away:
            continue
        upcoming.append(
            {
                "id": str(event.get("id")),
                "date": event.get("date") or "",
                "name": event.get("name") or f"{away['name']} at {home['name']}",
                "neutral": bool(comp.get("neutralSite")),
                "home": home,
                "away": away,
            }
        )
    upcoming.sort(key=lambda match: match["date"])
    season = payload.get("season") or {}
    year = season.get("year")
    if year is None:
        leagues = payload.get("leagues") or []
        if leagues:
            year = (leagues[0].get("season") or {}).get("year")
    week = payload.get("week") or {}
    if week.get("number"):
        week_label = f"Week {week['number']}"
    else:
        week_label = ""
    return {
        "upcoming": upcoming,
        "in_progress": in_progress,
        "season": int(year) if year else None,
        "week": week_label,
    }


def solve_linear(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting."""
    n = len(rhs)
    augmented = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-10:
            raise ValueError("difference system is singular")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        for j in range(col, n + 1):
            augmented[col][j] /= scale
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0:
                continue
            for j in range(col, n + 1):
                augmented[row][j] -= factor * augmented[col][j]
    return [augmented[i][n] for i in range(n)]


def game_weight(season: int, current: int, half_life: float = HALF_LIFE_SEASONS) -> float:
    age = max(0, current - int(season))
    return 0.5 ** (age / half_life)


def rate_games(games: list[dict], half_life: float = HALF_LIFE_SEASONS) -> dict:
    """Least-squares team ratings from home-score minus away-score."""
    if not games:
        raise ValueError("no played games")
    current = max(int(game["season"]) for game in games)
    team_ids = sorted({game["home_id"] for game in games} | {game["away_id"] for game in games})
    index = {team_id: i for i, team_id in enumerate(team_ids)}
    n = len(team_ids)
    dim = n + 1
    home_col = n
    matrix = [[0.0] * dim for _ in range(dim)]
    rhs = [0.0] * dim
    weight_total = 0.0
    points_total = 0.0
    for game in games:
        weight = game_weight(game["season"], current, half_life)
        home_i = index[game["home_id"]]
        away_i = index[game["away_id"]]
        margin = game["home_score"] - game["away_score"]
        matrix[home_i][home_i] += weight
        matrix[away_i][away_i] += weight
        matrix[home_i][away_i] -= weight
        matrix[away_i][home_i] -= weight
        rhs[home_i] += weight * margin
        rhs[away_i] -= weight * margin
        if not game["neutral"]:
            matrix[home_i][home_col] += weight
            matrix[home_col][home_i] += weight
            matrix[away_i][home_col] -= weight
            matrix[home_col][away_i] -= weight
            matrix[home_col][home_col] += weight
            rhs[home_col] += weight * margin
        weight_total += weight
        points_total += weight * (game["home_score"] + game["away_score"])
    # Neutral-only samples never observe a venue edge. A tiny ridge keeps that
    # column solvable and leaves a real venue edge effectively unchanged.
    matrix[home_col][home_col] += 1e-6
    # Pin the ratings to sum to zero so the venue edge stays identifiable.
    last = n - 1
    for col in range(dim):
        matrix[last][col] = 0.0
    for col in range(n):
        matrix[last][col] = 1.0
    rhs[last] = 0.0
    solution = solve_linear(matrix, rhs)
    ratings = {team_id: solution[i] for team_id, i in index.items()}
    home_advantage = solution[home_col]
    error = 0.0
    weighed = 0.0
    for game in games:
        weight = game_weight(game["season"], current, half_life)
        predicted = ratings[game["home_id"]] - ratings[game["away_id"]]
        if not game["neutral"]:
            predicted += home_advantage
        miss = (game["home_score"] - game["away_score"]) - predicted
        error += weight * miss * miss
        weighed += weight
    rmse = math.sqrt(error / weighed) if weighed else 0.0
    avg_total = points_total / weight_total if weight_total else 44.0
    return {
        "ratings": ratings,
        "home_advantage": home_advantage,
        "rmse": rmse,
        "avg_total": avg_total,
        "current_season": current,
        "games": len(games),
    }


def project_score(margin: float, avg_total: float) -> tuple[int, int]:
    home = (avg_total + margin) / 2
    away = (avg_total - margin) / 2
    home_pts = int(round(home))
    away_pts = int(round(away))
    if margin > 0 and home_pts <= away_pts:
        home_pts = away_pts + 1
    elif margin < 0 and away_pts <= home_pts:
        away_pts = home_pts + 1
    elif margin == 0:
        mid = int(round((home_pts + away_pts) / 2))
        home_pts = away_pts = mid
    return max(home_pts, 0), max(away_pts, 0)


def _record(team_id: str, games: list[dict], season: int | None = None) -> dict:
    wins = losses = ties = 0
    for game in games:
        if season is not None and int(game["season"]) != season:
            continue
        if game["home_id"] == team_id:
            margin = game["home_score"] - game["away_score"]
        elif game["away_id"] == team_id:
            margin = game["away_score"] - game["home_score"]
        else:
            continue
        if margin > 0:
            wins += 1
        elif margin < 0:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "losses": losses, "ties": ties}


def record_label(record: dict) -> str:
    if record["ties"]:
        return f"{record['wins']}-{record['losses']}-{record['ties']}"
    return f"{record['wins']}-{record['losses']}"


def _win_probability(margin: float, rmse: float) -> float:
    if rmse <= 0:
        return 0.5
    z = (margin / rmse) / math.sqrt(2)
    probability = 0.5 * (1 + math.erf(z))
    return min(0.99, max(0.01, probability))


def _public_side(side: dict, catalog: dict, rating: float, games: list[dict], season: int) -> dict:
    known = catalog.get(side["id"], {})
    full = _record(side["id"], games)
    this_season = _record(side["id"], games, season)
    return {
        "id": side["id"],
        "name": known.get("name") or side.get("name") or side["id"],
        "abbreviation": known.get("abbreviation") or side.get("abbreviation") or "",
        "nickname": known.get("nickname") or side.get("name") or "",
        "color": known.get("color") or "c4a35a",
        "logo": known.get("logo") or "",
        "rating": round(rating, 2),
        "record": record_label(full),
        "season_record": record_label(this_season),
    }


def _winner_name(side: dict) -> str:
    nickname = side.get("nickname") or side.get("name") or "That side"
    return f"The {nickname}"


def predict_match(match: dict, rated: dict, games: list[dict], catalog: dict) -> dict:
    ratings = rated["ratings"]
    home_rating = ratings.get(match["home"]["id"], 0.0)
    away_rating = ratings.get(match["away"]["id"], 0.0)
    margin = home_rating - away_rating
    if not match.get("neutral"):
        margin += rated["home_advantage"]
    home_pts, away_pts = project_score(margin, rated["avg_total"])
    season = rated["current_season"]
    home = _public_side(match["home"], catalog, home_rating, games, season)
    away = _public_side(match["away"], catalog, away_rating, games, season)
    if margin < 0:
        winner, loser = away, home
    else:
        winner, loser = home, away
    p_home = _win_probability(margin, rated["rmse"])
    p_winner = p_home if winner["id"] == home["id"] else 1 - p_home
    sentence = f"{_winner_name(winner)} win."
    return {
        "id": match["id"],
        "date": match["date"],
        "name": match["name"],
        "neutral": bool(match.get("neutral")),
        "home": home,
        "away": away,
        "margin": round(margin, 2),
        "difference": round(abs(margin), 2),
        "rating_gap": round(home_rating - away_rating, 2),
        "venue_edge": 0.0 if match.get("neutral") else round(rated["home_advantage"], 2),
        "projected": {"home": home_pts, "away": away_pts},
        "winner": {
            "id": winner["id"],
            "name": winner["name"],
            "nickname": winner["nickname"],
            "abbreviation": winner["abbreviation"],
        },
        "loser": {
            "id": loser["id"],
            "name": loser["name"],
            "nickname": loser["nickname"],
            "abbreviation": loser["abbreviation"],
        },
        "win_probability": round(p_winner, 4),
        "tossup": abs(margin) < 0.5,
        "sentence": sentence,
    }


def head_to_head(games: list[dict], team_a: str, team_b: str, limit: int = 12) -> list[dict]:
    rows = [
        game
        for game in games
        if team_a in (game["home_id"], game["away_id"]) and team_b in (game["home_id"], game["away_id"])
    ]
    rows.sort(key=lambda game: game["date"], reverse=True)
    view = []
    for game in rows[:limit]:
        view.append(
            {
                "id": game["id"],
                "date": game["date"],
                "season": game["season"],
                "neutral": game["neutral"],
                "home_name": game["home_name"],
                "away_name": game["away_name"],
                "home_abbr": game["home_abbr"],
                "away_abbr": game["away_abbr"],
                "home_score": game["home_score"],
                "away_score": game["away_score"],
            }
        )
    return view


def team_games(team_id: str, games: list[dict]) -> list[dict]:
    rows = []
    for game in games:
        if game["home_id"] == team_id:
            points_for = game["home_score"]
            points_against = game["away_score"]
            opponent = game["away_name"]
            opponent_id = game["away_id"]
            opponent_abbr = game["away_abbr"]
            location = "neutral" if game["neutral"] else "home"
        elif game["away_id"] == team_id:
            points_for = game["away_score"]
            points_against = game["home_score"]
            opponent = game["home_name"]
            opponent_id = game["home_id"]
            opponent_abbr = game["home_abbr"]
            location = "neutral" if game["neutral"] else "away"
        else:
            continue
        if points_for > points_against:
            result = "W"
        elif points_for < points_against:
            result = "L"
        else:
            result = "T"
        rows.append(
            {
                "id": game["id"],
                "date": game["date"],
                "season": game["season"],
                "season_type": game["season_type"],
                "week": game["week"],
                "opponent_id": opponent_id,
                "opponent": opponent,
                "opponent_abbr": opponent_abbr,
                "location": location,
                "points_for": points_for,
                "points_against": points_against,
                "result": result,
            }
        )
    rows.sort(key=lambda row: (row["date"], row["id"]))
    return rows


def build_report(games: list[dict], scoreboard: dict, catalog: dict) -> dict:
    rated = rate_games(games)
    season = rated["current_season"]
    upcoming = [
        predict_match(match, rated, games, catalog) for match in scoreboard.get("upcoming") or []
    ]
    next_game = upcoming[0] if upcoming else None
    seasons = [int(game["season"]) for game in games]
    since = min(seasons) if seasons else FIRST_SEASON
    standings = []
    for team_id, team in catalog.items():
        rating = rated["ratings"].get(team_id, 0.0)
        full = _record(team_id, games)
        this_season = _record(team_id, games, season)
        standings.append(
            {
                "id": team_id,
                "name": team["name"],
                "abbreviation": team["abbreviation"],
                "nickname": team["nickname"],
                "color": team["color"],
                "logo": team["logo"],
                "rating": round(rating, 2),
                "record": record_label(full),
                "season_record": record_label(this_season),
                "games": full["wins"] + full["losses"] + full["ties"],
            }
        )
    standings.sort(key=lambda row: (-row["rating"], row["name"]))
    meetings = []
    if next_game:
        meetings = head_to_head(games, next_game["home"]["id"], next_game["away"]["id"])
    if next_game:
        call = next_game["sentence"]
        where = "on a neutral field" if next_game["neutral"] else (
            "at home" if next_game["winner"]["id"] == next_game["home"]["id"] else "on the road"
        )
        detail = (
            f"{next_game['away']['name']} at {next_game['home']['name']}. "
            f"{call[:-1]} {where}. The difference is {next_game['difference']:.1f} points, "
            f"from {len(games)} games already played since {since}."
        )
    else:
        call = "No game is left on the board."
        detail = "Every game on the current board has already kicked off."
    return {
        "sport": "NFL",
        "call": call,
        "detail": detail,
        "since_season": since,
        "through_season": season,
        "games_played": len(games),
        "teams": len(catalog) or len(rated["ratings"]),
        "home_advantage": round(rated["home_advantage"], 2),
        "rmse": round(rated["rmse"], 2),
        "half_life_seasons": HALF_LIFE_SEASONS,
        "method": METHOD,
        "in_progress": scoreboard.get("in_progress") or 0,
        "week": scoreboard.get("week") or "",
        "next_game": next_game,
        "upcoming": upcoming,
        "standings": standings,
        "meetings": meetings,
    }


def load_cache() -> dict:
    if not CACHE_PATH.is_file():
        return {}
    try:
        payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    seasons = payload.get("seasons") or {}
    return seasons if isinstance(seasons, dict) else {}


def save_cache(seasons: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CACHE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"version": 1, "seasons": seasons}), encoding="utf-8")
    temporary.replace(CACHE_PATH)


def _schedule_url(team_id: str, season: int, season_type: int) -> str:
    return SCHEDULE_URL.format(team_id=team_id, season=season, season_type=season_type)


def fetch_season(season: int, season_type: int, team_ids: list[str], workers: int = 12) -> tuple[list[dict], int]:
    ok = 0
    collected: list[dict] = []

    def one(team_id: str):
        try:
            return fetch_json(_schedule_url(team_id, season, season_type))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for payload in pool.map(one, team_ids):
            if payload is None:
                continue
            ok += 1
            collected.extend(games_from_schedule(payload))
    return merge_games(collected), ok


def ensure_games(catalog: dict, current_season: int, refresh: bool = False) -> list[dict]:
    """Load played games, fetching any season not already cached.

    Completed seasons stay on disk. The current season is always reread.
    """
    team_ids = list(catalog)
    cached = {} if refresh else load_cache()
    changed = False
    for year in range(FIRST_SEASON, current_season + 1):
        key = str(year)
        if key in cached and year != current_season and not refresh:
            continue
        print(f"Reading {year} games already played…", flush=True)
        regular, regular_ok = fetch_season(year, 2, team_ids)
        post, post_ok = fetch_season(year, 3, team_ids)
        if regular_ok == 0 and post_ok == 0:
            print(f"Skipped {year}; the results feed did not answer.", flush=True)
            continue
        cached[key] = merge_games(regular, post)
        changed = True
        save_cache(cached)
        print(f"  {year}: {len(cached[key])} games", flush=True)
    games = []
    for key in sorted(cached, key=lambda item: int(item)):
        games.extend(cached[key])
    return merge_games(games)


def get_report(refresh: bool = False) -> dict:
    with _lock:
        if _memory["report"] is not None and not refresh:
            return _memory["report"]
        catalog = parse_teams(fetch_json(TEAMS_URL))
        board = parse_scoreboard(fetch_json(SCOREBOARD_URL))
        current = board.get("season") or max(FIRST_SEASON, 2026)
        games = ensure_games(catalog, current, refresh=refresh)
        report = build_report(games, board, catalog)
        _memory["report"] = report
        _memory["games"] = games
        _memory["teams"] = catalog
        return report


def team_history(team_id: str) -> dict | None:
    with _lock:
        if _memory["games"] is None:
            need = True
        else:
            need = False
            games = _memory["games"]
            catalog = _memory["teams"]
    if need:
        get_report()
        games = _memory["games"]
        catalog = _memory["teams"]
    team = (catalog or {}).get(str(team_id))
    if team is None:
        return None
    return {"team": team, "games": team_games(str(team_id), games or [])}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Football and boxing difference engine")
    parser.add_argument("--refresh", action="store_true", help="reread every season")
    parser.add_argument("--print", dest="show", action="store_true", help="print the next-game call")
    parser.add_argument("--serve", action="store_true", help="serve the presentation")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    report = get_report(refresh=args.refresh)
    print("Football:", report["call"], flush=True)
    print(report["detail"], flush=True)
    try:
        from boxing import get_boxing_report

        boxing_report = get_boxing_report(refresh=args.refresh)
        print("Boxing:", boxing_report["call"], flush=True)
        print(boxing_report["detail"], flush=True)
    except Exception as exc:
        print(f"Boxing history unavailable: {exc}", flush=True)
    if args.show and not args.serve:
        return
    from flask import Flask, jsonify, send_from_directory

    app = Flask(__name__, static_folder=str(ROOT / "static"), static_url_path="/static")

    @app.get("/")
    @app.get("/football")
    @app.get("/boxing")
    def page():
        return send_from_directory(ROOT / "static", "football.html")

    @app.get("/api/football")
    def api():
        return jsonify(get_report())

    @app.get("/api/football/teams/<team_id>")
    def api_team(team_id: str):
        payload = team_history(team_id)
        if payload is None:
            return jsonify({"error": "unknown team"}), 404
        return jsonify(payload)

    @app.get("/api/boxing")
    def api_boxing():
        from boxing import get_boxing_report

        return jsonify(get_boxing_report())

    print(f"Open http://127.0.0.1:{args.port}/football", flush=True)
    try:
        from waitress import serve as waitress_serve
    except ImportError:
        app.run(host=args.host, port=args.port, threaded=True)
    else:
        waitress_serve(app, host=args.host, port=args.port, threads=4)


if __name__ == "__main__":
    main()
