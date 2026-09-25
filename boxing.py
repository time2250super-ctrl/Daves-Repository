#!/usr/bin/env python3
"""Boxing side of the difference engine.

Each finished bout is a result difference: a stoppage is a wider gap than a
decision, and a draw is no gap. The ring is neutral. The same least-squares
fit used for football rates the fighters whose records meet, then names the
winner of the next bout on the schedule.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import threading
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

from difference_engine import HALF_LIFE_SEASONS, fetch_json, rate_games

ROOT = Path(__file__).resolve().parent
CACHE_PATH = Path(os.environ.get("BOXING_CACHE", str(ROOT / "data" / "boxing_cache.json")))
SCHEDULE_URL = "https://www.espn.com/boxing/story/_/id/12508267/boxing-fight-schedule"
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_UA = "DifferenceEngine/1.0 (local sports history)"
# A full browser UA is challenged; the short token is what this host accepts.
BROWSER_UA = "Mozilla/5.0"

METHOD = (
    "Each finished bout is a result difference: a stoppage is a wider gap than "
    "a decision, and a draw is no gap. The ring is neutral, so there is no home "
    "edge. A bout from four years ago counts half as much as one from this year. "
    "The engine solves those differences for the fighters whose records meet, "
    "then applies them to the next bout."
)

# Wider gaps for finishes, smaller gaps for close decisions.
MARGINS = {
    "KO": 10,
    "TKO": 10,
    "RTD": 8,
    "UD": 6,
    "PTS": 6,
    "MD": 5,
    "TD": 4,
    "DQ": 4,
    "SD": 3,
}

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
DATE_LINE = re.compile(
    r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+(\d{1,2}):\s*(.*)$",
    re.I,
)
VS_LINE = re.compile(r"^(?:Title fight:\s*)?(.+?)\s+vs\.?\s+(.+?)(?:,\s*(.*))?$", re.I)
BOUT_DATE = re.compile(
    r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    r"\s+(\d{4})",
    re.I,
)
LINK = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
MONTH_NAMES = (
    "January|February|March|April|May|June|July|August|September|"
    "October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
BOUT_DATE_MDY = re.compile(rf"({MONTH_NAMES})\s+(\d{{1,2}}),\s+(\d{{4}})", re.I)

_lock = threading.Lock()
_memory: dict = {"report": None}


def canon(name: str) -> str:
    base = name.split("(")[0]
    base = unicodedata.normalize("NFKD", base)
    base = "".join(ch for ch in base if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]", "", base.lower())


def fetch_html(url: str) -> str:
    """ESPN's edge blocks bare urllib. curl gets the schedule page through."""
    completed = subprocess.run(
        ["curl", "-fsSL", "--max-time", "30", "-A", BROWSER_UA, url],
        check=True,
        capture_output=True,
    )
    return completed.stdout.decode("utf-8", "replace")


def html_to_lines(html: str) -> list[str]:
    start = html.find("Full schedule")
    if start < 0:
        start = 0
    end = html.find("Terms of Use", start)
    chunk = html[start:end if end > start else None]
    text = re.sub(r"<br\s*/?>", "\n", chunk, flags=re.I)
    text = re.sub(r"</(p|h2|h3|li|div|tr)>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    lines = []
    for line in text.splitlines():
        clean = re.sub(r"\s+", " ", line).strip()
        if clean:
            lines.append(clean)
    return lines


def parse_schedule(html: str, today: date | None = None) -> list[dict]:
    """Upcoming bouts in schedule order. A month that wraps backward starts the next year."""
    today = today or datetime.now(timezone.utc).date()
    year = today.year
    previous_month = None
    current = None
    place = ""
    bouts = []
    for line in html_to_lines(html):
        dated = DATE_LINE.match(line)
        if dated:
            month = MONTHS[dated.group(1).lower()]
            day = int(dated.group(2))
            if previous_month is not None and month < previous_month:
                year += 1
            previous_month = month
            current = date(year, month, day)
            place = dated.group(3).strip()
            continue
        if current is None or current < today or " vs" not in line.lower():
            continue
        matched = VS_LINE.match(line)
        if not matched:
            continue
        left = _clean_name(matched.group(1))
        right = _clean_name(matched.group(2))
        if not left or not right or "tba" in (left.lower(), right.lower()):
            continue
        bouts.append(
            {
                "date": current.isoformat(),
                "place": place,
                "title": line.lower().startswith("title fight"),
                "a": left,
                "b": right,
                "note": (matched.group(3) or "").strip(),
            }
        )
    return bouts


def _clean_name(name: str) -> str:
    name = re.sub(r"^Title fight:\s*", "", name, flags=re.I).strip(" -–")
    name = name.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", name).strip()


def parse_bouts(wikitext: str, fighter_title: str) -> list[dict]:
    """Finished professional bouts from a Wikipedia record table."""
    if not wikitext:
        return []
    lowered = wikitext.lower()
    start = lowered.find("professional boxing record")
    if start < 0:
        start = lowered.find("professional record")
    if start < 0:
        return []
    body = wikitext[start:]
    fighter_id = canon(fighter_title)
    display = fighter_title.split("(")[0].strip()
    bouts = []
    for part in re.split(r"\n\|-", body):
        cells = [line[1:].strip() for line in part.splitlines() if line.startswith("|")]
        if len(cells) < 7:
            continue
        result = _result(cells[1])
        if result is None:
            continue
        opponent_title, opponent_name = _opponent(cells[3])
        if not opponent_title:
            continue
        method = _method(cells[4])
        when = _bout_date(cells[6])
        if when is None:
            continue
        opponent_id = canon(opponent_title)
        if not opponent_id or opponent_id == fighter_id:
            continue
        bouts.append(
            {
                "date": when,
                "year": int(when[:4]),
                "result": result,
                "method": method,
                "fighter_id": fighter_id,
                "fighter": display,
                "opponent_id": opponent_id,
                "opponent": opponent_name,
                "opponent_title": opponent_title,
            }
        )
    bouts.sort(key=lambda bout: bout["date"])
    return bouts


def _result(cell: str) -> str | None:
    text = re.sub(r"\{\{[^}]+\}\}", " ", cell)
    text = re.sub(r"[^A-Za-z]", " ", text).lower()
    if "draw" in text:
        return "D"
    if re.search(r"\bnc\b", text) or "no contest" in text:
        return None
    if "loss" in text:
        return "L"
    if "win" in text:
        return "W"
    return None


def _opponent(cell: str) -> tuple[str, str]:
    matched = LINK.search(cell)
    if not matched:
        plain = re.sub(r"\{\{[^}]+\}\}", "", cell)
        plain = re.sub(r"<[^>]+>", "", plain)
        plain = re.sub(r'^(?:align=left\||style="[^"]*"\|)+', "", plain).strip()
        return plain, plain
    title = matched.group(1).strip()
    label = (matched.group(2) or title).strip()
    label = re.sub(r"\{\{[^}]+\}\}", "", label).strip()
    return title, label or title


def _method(cell: str) -> str:
    token = re.sub(r"\{\{[^}]+\}\}", " ", cell)
    token = re.sub(r"[^A-Za-z]", " ", token).strip().split()
    if not token:
        return "PTS"
    code = token[0].upper()
    return code if code in MARGINS else "PTS"


def _month_number(token: str) -> int:
    key = token.lower()
    if key.startswith("sept"):
        return 9
    return MONTHS.get(key[:3], 0)


def _bout_date(cell: str) -> str | None:
    matched = BOUT_DATE.search(cell)
    if matched:
        day = int(matched.group(1))
        month = _month_number(matched.group(2))
        year = int(matched.group(3))
    else:
        matched = BOUT_DATE_MDY.search(cell)
        if not matched:
            return None
        month = _month_number(matched.group(1))
        day = int(matched.group(2))
        year = int(matched.group(3))
    if not month:
        return None
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def bouts_to_games(bouts: list[dict]) -> list[dict]:
    """One neutral result per bout. The winner is stored on the home side so the margin is positive."""
    games = {}
    for bout in bouts:
        margin = 0 if bout["result"] == "D" else MARGINS.get(bout["method"], 6)
        if bout["result"] == "L":
            home_id, away_id = bout["opponent_id"], bout["fighter_id"]
            home_name, away_name = bout["opponent"], bout["fighter"]
        else:
            home_id, away_id = bout["fighter_id"], bout["opponent_id"]
            home_name, away_name = bout["fighter"], bout["opponent"]
        if bout["result"] == "D":
            home_score = away_score = 0
        else:
            home_score, away_score = margin, 0
        bout_id = "|".join(sorted((home_id, away_id)) + [bout["date"]])
        games[bout_id] = {
            "id": bout_id,
            "season": bout["year"],
            "season_type": 2,
            "week": None,
            "date": bout["date"] + "T00:00:00Z",
            "neutral": True,
            "home_id": home_id,
            "away_id": away_id,
            "home_name": home_name,
            "away_name": away_name,
            "home_abbr": "",
            "away_abbr": "",
            "home_score": home_score,
            "away_score": away_score,
        }
    return list(games.values())


def _adjacent(games: list[dict]) -> dict[str, set[str]]:
    adjacent: dict[str, set[str]] = defaultdict(set)
    for game in games:
        adjacent[game["home_id"]].add(game["away_id"])
        adjacent[game["away_id"]].add(game["home_id"])
    return adjacent


def same_component(games: list[dict], left: str, right: str) -> bool:
    if left == right:
        return True
    adjacent = _adjacent(games)
    if left not in adjacent or right not in adjacent:
        return False
    seen = {left}
    stack = [left]
    while stack:
        node = stack.pop()
        if node == right:
            return True
        for other in adjacent[node]:
            if other not in seen:
                seen.add(other)
                stack.append(other)
    return False


def _win_probability(margin: float, rmse: float) -> float:
    if rmse <= 0:
        return 0.5
    probability = 0.5 * (1 + math.erf((margin / rmse) / math.sqrt(2)))
    return min(0.99, max(0.01, probability))


def record_strength(bouts: list[dict], current_year: int) -> float | None:
    weighted = 0.0
    total = 0.0
    for bout in bouts:
        weight = 0.5 ** (max(0, current_year - bout["year"]) / HALF_LIFE_SEASONS)
        if bout["result"] == "W":
            value = 1.0
        elif bout["result"] == "L":
            value = 0.0
        else:
            value = 0.5
        weighted += weight * value
        total += weight
    if total == 0:
        return None
    return weighted / total


def record_label(bouts: list[dict]) -> str:
    wins = sum(1 for bout in bouts if bout["result"] == "W")
    losses = sum(1 for bout in bouts if bout["result"] == "L")
    draws = sum(1 for bout in bouts if bout["result"] == "D")
    if draws:
        return f"{wins}-{losses}-{draws}"
    return f"{wins}-{losses}"


def _history_rows(bouts: list[dict]) -> list[dict]:
    rows = []
    for bout in bouts:
        rows.append(
            {
                "date": bout["date"],
                "opponent": bout["opponent"],
                "result": bout["result"],
                "method": bout["method"],
            }
        )
    return rows


def component_games(games: list[dict], seed: str) -> list[dict]:
    adjacent = _adjacent(games)
    if seed not in adjacent:
        return []
    seen = set()
    stack = [seed]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adjacent[node] - seen)
    return [game for game in games if game["home_id"] in seen and game["away_id"] in seen]


def decide_bout(games: list[dict], left: dict, right: dict, left_bouts: list[dict], right_bouts: list[dict]) -> dict:
    """Name a winner from the shared graph, or from each fighter's own record."""
    linked = same_component(games, left["id"], right["id"])
    current_year = max((game["season"] for game in games), default=date.today().year)
    rated = None
    if linked and games:
        try:
            rated = rate_games(component_games(games, left["id"]))
        except ValueError:
            rated = None
            linked = False
    if rated is not None:
        margin = rated["ratings"].get(left["id"], 0.0) - rated["ratings"].get(right["id"], 0.0)
        rmse = rated["rmse"]
        basis = "shared history"
        left_rating = round(rated["ratings"].get(left["id"], 0.0), 2)
        right_rating = round(rated["ratings"].get(right["id"], 0.0), 2)
    else:
        left_strength = record_strength(left_bouts, current_year)
        right_strength = record_strength(right_bouts, current_year)
        # An absent record is an even fighter. Played results move the number off 0.5.
        left_value = 0.5 if left_strength is None else left_strength
        right_value = 0.5 if right_strength is None else right_strength
        margin = (left_value - right_value) * 10
        rmse = 4.0
        basis = "own records"
        left_rating = None if left_strength is None else round(left_value, 2)
        right_rating = None if right_strength is None else round(right_value, 2)
    if margin > 0 or (margin == 0 and left["name"] <= right["name"]):
        winner, loser = left, right
        winner_margin = margin
    else:
        winner, loser = right, left
        winner_margin = -margin
    probability = _win_probability(abs(winner_margin), rmse if rmse else 4.0)
    if abs(margin) < 0.35:
        probability = max(probability, 0.5)
    return {
        "winner": winner,
        "loser": loser,
        "margin": round(margin, 2),
        "difference": round(abs(margin), 2),
        "win_probability": round(probability, 4),
        "tossup": abs(margin) < 0.35,
        "basis": basis,
        "left_rating": left_rating,
        "right_rating": right_rating,
        "sentence": f"{winner['name']} wins.",
    }


def build_boxing_report(bouts_on_schedule: list[dict], pages: dict[str, str], resolved: dict[str, str]) -> dict:
    """pages maps Wikipedia title to wikitext. resolved maps schedule spelling to that title."""
    records: dict[str, dict[tuple, dict]] = {}
    names: dict[str, str] = {}

    def add_page(title: str) -> None:
        wikitext = pages.get(title) or ""
        if not wikitext:
            return
        for bout in parse_bouts(wikitext, title):
            bucket = records.setdefault(bout["fighter_id"], {})
            bucket[(bout["date"], bout["opponent_id"])] = bout
            names.setdefault(bout["fighter_id"], bout["fighter"])
            names.setdefault(bout["opponent_id"], bout["opponent"])

    for spelling, title in resolved.items():
        fighter_id = canon(title or spelling)
        names.setdefault(fighter_id, (title or spelling).split("(")[0].strip())
        if title:
            add_page(title)
    for title in pages:
        add_page(title)
    records = {fighter_id: list(bucket.values()) for fighter_id, bucket in records.items()}
    all_bouts = [bout for group in records.values() for bout in group]
    games = bouts_to_games(all_bouts)

    def side(spelling: str) -> dict:
        title = resolved.get(spelling) or ""
        fighter_id = canon(title or spelling)
        own = [bout for bout in records.get(fighter_id, []) if bout["fighter_id"] == fighter_id]
        return {
            "id": fighter_id,
            "name": names.get(fighter_id) or spelling,
            "record": record_label(own) if own else "no record",
            "bouts": _history_rows(own),
            "found": bool(own),
        }

    predictions = []
    for bout in bouts_on_schedule:
        left = side(bout["a"])
        right = side(bout["b"])
        left_bouts = records.get(left["id"], [])
        right_bouts = records.get(right["id"], [])
        decision = decide_bout(games, left, right, left_bouts, right_bouts)
        predictions.append(
            {
                "date": bout["date"],
                "place": bout["place"],
                "title": bout["title"],
                "note": bout.get("note") or "",
                "left": {key: left[key] for key in ("id", "name", "record", "found")},
                "right": {key: right[key] for key in ("id", "name", "record", "found")},
                "left_rating": decision["left_rating"],
                "right_rating": decision["right_rating"],
                "margin": decision["margin"],
                "difference": decision["difference"],
                "win_probability": decision["win_probability"],
                "tossup": decision["tossup"],
                "basis": decision["basis"],
                "winner": {"id": decision["winner"]["id"], "name": decision["winner"]["name"]},
                "loser": {"id": decision["loser"]["id"], "name": decision["loser"]["name"]},
                "sentence": decision["sentence"],
            }
        )
    next_bout = predictions[0] if predictions else None
    if next_bout:
        left = side(bouts_on_schedule[0]["a"])
        right = side(bouts_on_schedule[0]["b"])
        call = next_bout["sentence"]
        how = (
            "from bouts that connect them"
            if next_bout["basis"] == "shared history"
            else "from each fighter's own wins and losses"
        )
        detail = (
            f"{next_bout['left']['name']} vs. {next_bout['right']['name']} "
            f"on {next_bout['date']}"
            + (f" in {next_bout['place']}" if next_bout["place"] else "")
            + f". {call[:-1]} {how}. The difference is {next_bout['difference']:.1f}."
        )
        history = {"left": left["bouts"], "right": right["bouts"]}
    else:
        call = "No bout is left on the schedule."
        detail = "The schedule has no upcoming bout with both fighters named."
        history = {"left": [], "right": []}
    years = [int(bout["date"][:4]) for bout in all_bouts]
    return {
        "sport": "Boxing",
        "call": call,
        "detail": detail,
        "method": METHOD,
        "games_played": len(games),
        "since_season": min(years) if years else None,
        "through_season": max(years) if years else None,
        "next_bout": next_bout,
        "upcoming": predictions[1:],
        "history": history,
    }


def _wiki_search(name: str) -> str | None:
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": f"{name} boxer",
            "srlimit": "3",
            "format": "json",
        }
    )
    request = urllib.request.Request(f"{WIKI_API}?{query}", headers={"User-Agent": WIKI_UA})
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            payload = json.loads(response.read().decode())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None
    wanted = canon(name)
    for hit in (payload.get("query") or {}).get("search") or []:
        title = hit.get("title") or ""
        if canon(title) == wanted:
            return title
    return None


def _wiki_wikitext(title: str) -> str:
    query = urllib.parse.urlencode(
        {"action": "parse", "page": title, "prop": "wikitext", "format": "json", "redirects": "1"}
    )
    request = urllib.request.Request(f"{WIKI_API}?{query}", headers={"User-Agent": WIKI_UA})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode())
    return ((payload.get("parse") or {}).get("wikitext") or {}).get("*") or ""


def load_cache() -> dict:
    if not CACHE_PATH.is_file():
        return {"pages": {}}
    try:
        payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"pages": {}}
    payload.setdefault("pages", {})
    return payload


def save_cache(payload: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CACHE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(CACHE_PATH)


def _ensure_page(title: str, cache: dict) -> None:
    if not title or title in cache["pages"]:
        return
    try:
        cache["pages"][title] = _wiki_wikitext(title)
        print(f"Read record for {title}", flush=True)
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, KeyError):
        cache["pages"][title] = ""


def get_boxing_report(refresh: bool = False) -> dict:
    with _lock:
        if _memory["report"] is not None and not refresh:
            return _memory["report"]
        html = fetch_html(SCHEDULE_URL)
        upcoming = parse_schedule(html)
        # The next card, plus later title fights, is the board we actually call.
        if not upcoming:
            report = build_boxing_report([], {}, {})
            _memory["report"] = report
            return report
        next_date = upcoming[0]["date"]
        next_place = upcoming[0]["place"]
        card = [
            bout for bout in upcoming
            if bout["date"] == next_date and bout["place"] == next_place
        ][:6]
        titles = [bout for bout in upcoming if bout["title"] and bout not in card][:6]
        seen = set()
        selected = []
        for bout in card + titles:
            key = (bout["date"], canon(bout["a"]), canon(bout["b"]))
            if key in seen:
                continue
            seen.add(key)
            selected.append(bout)
        cache = {"pages": {}} if refresh else load_cache()
        resolved: dict[str, str] = {}
        for bout in selected:
            for spelling in (bout["a"], bout["b"]):
                if spelling in resolved:
                    continue
                title = _wiki_search(spelling)
                resolved[spelling] = title or ""
                if title:
                    _ensure_page(title, cache)
        # If the headline pair never meets, pull recent opponents into the graph.
        headline = selected[0]
        left_title = resolved.get(headline["a"]) or ""
        right_title = resolved.get(headline["b"]) or ""
        left_bouts = parse_bouts(cache["pages"].get(left_title, ""), left_title) if left_title else []
        right_bouts = parse_bouts(cache["pages"].get(right_title, ""), right_title) if right_title else []
        games = bouts_to_games(left_bouts + right_bouts)
        left_id = canon(left_title or headline["a"])
        right_id = canon(right_title or headline["b"])
        if left_bouts and right_bouts and not same_component(games, left_id, right_id):
            opponents = []
            for bout in reversed(left_bouts + right_bouts):
                if bout["opponent_title"] not in opponents:
                    opponents.append(bout["opponent_title"])
            for title in opponents[:16]:
                _ensure_page(title, cache)
        save_cache(cache)
        report = build_boxing_report(selected, cache["pages"], resolved)
        _memory["report"] = report
        return report
