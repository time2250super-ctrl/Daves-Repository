"""The difference engine rates teams from games already played and names a winner."""

from __future__ import annotations

import difference_engine as engine


def _game(gid, season, home, away, home_score, away_score, neutral=False, date=None):
    return {
        "id": gid,
        "season": season,
        "season_type": 2,
        "week": 1,
        "date": date or f"{season}-09-01T17:00Z",
        "neutral": neutral,
        "home_id": home,
        "away_id": away,
        "home_name": home,
        "away_name": away,
        "home_abbr": home[:3].upper(),
        "away_abbr": away[:3].upper(),
        "home_score": home_score,
        "away_score": away_score,
    }


def _final_event(home_name, away_name, home_score, away_score, year=2024):
    return {
        "id": "401",
        "date": f"{year}-09-08T00:20Z",
        "name": f"{away_name} at {home_name}",
        "season": {"year": year},
        "seasonType": {"type": 2, "name": "Regular Season"},
        "week": {"number": 1},
        "competitions": [
            {
                "neutralSite": False,
                "status": {"type": {"state": "post", "completed": True, "name": "STATUS_FINAL"}},
                "competitors": [
                    {
                        "homeAway": "home",
                        "score": {"value": float(home_score), "displayValue": str(home_score)},
                        "team": {"id": "2", "displayName": home_name, "abbreviation": "HOM"},
                    },
                    {
                        "homeAway": "away",
                        "score": {"value": float(away_score), "displayValue": str(away_score)},
                        "team": {"id": "33", "displayName": away_name, "abbreviation": "AWY"},
                    },
                ],
            }
        ],
    }


def test_schedule_parser_keeps_finished_games_only():
    scheduled = _final_event("Buffalo Bills", "Baltimore Ravens", 41, 40)
    scheduled["id"] = "open"
    scheduled["competitions"][0]["status"] = {
        "type": {"state": "pre", "completed": False, "name": "STATUS_SCHEDULED"}
    }
    scheduled["competitions"][0]["competitors"][0]["score"] = None
    scheduled["competitions"][0]["competitors"][1]["score"] = None
    final = _final_event("Buffalo Bills", "Baltimore Ravens", 41, 40)
    games = engine.games_from_schedule({"events": [scheduled, final]})
    assert len(games) == 1
    assert games[0]["home_score"] == 41
    assert games[0]["away_score"] == 40
    assert games[0]["home_id"] == "2"
    assert games[0]["season"] == 2024
    assert games[0]["season_type"] == 2


def test_merge_games_counts_a_result_once():
    first = _game("1", 2024, "A", "B", 20, 10)
    second = dict(first)
    second["home_name"] = "Home copy"
    assert len(engine.merge_games([first], [second])) == 1


def test_neutral_results_order_teams_by_score_difference():
    games = [
        _game("1", 2024, "A", "B", 10, 0, neutral=True),
        _game("2", 2024, "B", "C", 10, 0, neutral=True),
        _game("3", 2024, "A", "C", 20, 0, neutral=True),
    ]
    rated = engine.rate_games(games)
    ratings = rated["ratings"]
    assert ratings["A"] > ratings["B"] > ratings["C"]
    assert abs((ratings["A"] - ratings["B"]) - 10) < 0.25
    assert abs(rated["home_advantage"]) < 0.25


def test_home_and_away_splits_identify_the_venue_edge():
    games = [
        _game("1", 2024, "A", "B", 17, 7),
        _game("2", 2024, "B", "A", 17, 7),
    ]
    rated = engine.rate_games(games)
    assert abs(rated["ratings"]["A"] - rated["ratings"]["B"]) < 0.25
    assert abs(rated["home_advantage"] - 10) < 0.25


def test_older_season_weighs_less_than_this_season():
    recent = engine.game_weight(2026, 2026, half_life=10)
    decade = engine.game_weight(2016, 2026, half_life=10)
    assert recent == 1
    assert abs(decade - 0.5) < 1e-9
    assert engine.game_weight(2022, 2026) < engine.game_weight(2025, 2026) < 1


def test_next_game_goes_to_the_stronger_side():
    games = [
        _game("1", 2024, "A", "B", 10, 0, neutral=True),
        _game("2", 2024, "B", "C", 10, 0, neutral=True),
        _game("3", 2024, "A", "C", 20, 0, neutral=True),
    ]
    catalog = {
        "A": {"id": "A", "name": "Alpha", "abbreviation": "ALP", "nickname": "Alphas", "color": "112233", "logo": ""},
        "B": {"id": "B", "name": "Beta", "abbreviation": "BET", "nickname": "Betas", "color": "445566", "logo": ""},
        "C": {"id": "C", "name": "Gamma", "abbreviation": "GAM", "nickname": "Gammas", "color": "778899", "logo": ""},
    }
    board = {
        "upcoming": [
            {
                "id": "next",
                "date": "2026-09-27T17:00Z",
                "name": "Beta at Alpha",
                "neutral": True,
                "home": {"id": "A", "name": "Alpha", "abbreviation": "ALP"},
                "away": {"id": "B", "name": "Beta", "abbreviation": "BET"},
            }
        ],
        "in_progress": 1,
        "season": 2026,
        "week": "Week 3",
    }
    # The rating sample is 2024, so mark a later season on a copy for "current".
    # build_report takes current season from the games themselves.
    report = engine.build_report(games, board, catalog)
    assert report["next_game"]["winner"]["nickname"] == "Alphas"
    assert report["call"] == "The Alphas win."
    assert report["games_played"] == 3
    assert report["in_progress"] == 1
    assert report["upcoming"][0]["projected"]["home"] > report["upcoming"][0]["projected"]["away"]
    assert report["standings"][0]["id"] == "A"
    assert [row["id"] for row in report["meetings"]] == ["1"]


def test_played_meetings_are_listed_for_the_next_game():
    games = [
        _game("old", 2020, "A", "B", 24, 17, date="2020-10-01T17:00Z"),
        _game("new", 2024, "B", "A", 21, 20, date="2024-11-01T17:00Z"),
    ]
    catalog = {
        "A": {"id": "A", "name": "Alpha", "abbreviation": "ALP", "nickname": "Alphas", "color": "112233", "logo": ""},
        "B": {"id": "B", "name": "Beta", "abbreviation": "BET", "nickname": "Betas", "color": "445566", "logo": ""},
    }
    board = {
        "upcoming": [
            {
                "id": "next",
                "date": "2026-09-27T17:00Z",
                "name": "Beta at Alpha",
                "neutral": False,
                "home": {"id": "A", "name": "Alpha", "abbreviation": "ALP"},
                "away": {"id": "B", "name": "Beta", "abbreviation": "BET"},
            }
        ],
        "in_progress": 0,
        "week": "",
    }
    report = engine.build_report(games, board, catalog)
    assert [row["id"] for row in report["meetings"]] == ["new", "old"]
    history = engine.team_games("A", games)
    assert [row["result"] for row in history] == ["W", "L"]
    assert history[0]["opponent"] == "B"
    assert history[1]["location"] == "away"


def test_scoreboard_picks_the_earliest_game_not_yet_started():
    payload = {
        "season": {"year": 2026},
        "week": {"number": 3},
        "events": [
            {
                "id": "live",
                "date": "2026-09-25T00:15Z",
                "name": "Live game",
                "competitions": [
                    {
                        "neutralSite": False,
                        "status": {"type": {"state": "in", "completed": False}},
                        "competitors": [],
                    }
                ],
            },
            {
                "id": "later",
                "date": "2026-09-28T17:00Z",
                "name": "Later",
                "competitions": [
                    {
                        "neutralSite": False,
                        "status": {"type": {"state": "pre", "completed": False}},
                        "competitors": [
                            {"homeAway": "home", "team": {"id": "9", "displayName": "Home Late", "abbreviation": "H"}},
                            {"homeAway": "away", "team": {"id": "1", "displayName": "Away Late", "abbreviation": "A"}},
                        ],
                    }
                ],
            },
            {
                "id": "next",
                "date": "2026-09-27T17:00Z",
                "name": "Next",
                "competitions": [
                    {
                        "neutralSite": False,
                        "status": {"type": {"state": "pre", "completed": False}},
                        "competitors": [
                            {"homeAway": "home", "team": {"id": "2", "displayName": "Buffalo Bills", "abbreviation": "BUF"}},
                            {"homeAway": "away", "team": {"id": "24", "displayName": "Los Angeles Chargers", "abbreviation": "LAC"}},
                        ],
                    }
                ],
            },
            {
                "id": "done",
                "date": "2026-09-20T17:00Z",
                "name": "Done",
                "competitions": [
                    {
                        "neutralSite": False,
                        "status": {"type": {"state": "post", "completed": True}},
                        "competitors": [],
                    }
                ],
            },
        ],
    }
    board = engine.parse_scoreboard(payload)
    assert board["season"] == 2026
    assert board["in_progress"] == 1
    assert board["week"] == "Week 3"
    assert [match["id"] for match in board["upcoming"]] == ["next", "later"]


def test_project_score_keeps_the_winner():
    home, away = engine.project_score(0.4, 44)
    assert home > away
    home, away = engine.project_score(-3, 41)
    assert away > home


def test_football_routes(monkeypatch):
    import serve

    sample = {"call": "The Bills win.", "detail": "from games already played", "next_game": None, "upcoming": []}
    monkeypatch.setattr(engine, "get_report", lambda refresh=False: sample)
    monkeypatch.setattr(engine, "team_history", lambda team_id: {"team": {"id": team_id}, "games": []} if team_id == "2" else None)
    serve.app.config.update(TESTING=True)
    client = serve.app.test_client()
    page = client.get("/football")
    assert page.status_code == 200
    assert b"Difference Engine" in page.data
    api = client.get("/api/football")
    assert api.status_code == 200
    assert api.get_json()["call"] == "The Bills win."
    history = client.get("/api/football/teams/2")
    assert history.status_code == 200
    assert history.get_json()["team"]["id"] == "2"
    missing = client.get("/api/football/teams/nope")
    assert missing.status_code == 404
