"""Boxing calls come from bouts already fought."""

from __future__ import annotations

from datetime import date

import boxing


FUNDORA = """
==Professional boxing record==
{|class=wikitable
|-
|2
|{{yes2}}Win
|2–0
|align=left|[[Tim Tszyu]]
|TKO
|7 (12)
|19 Jul 2025
|align=left|Las Vegas
|-
|1
|{{no2}}Loss
|1–1
|align=left|[[Brian Mendoza]]
|KO
|7 (12)
|8 Apr 2023
|align=left|Carson
|-
|0
|{{draw}}Draw
|0–0–1
|align=left|[[Carlos Ocampo]]
|SD
|12
|8 Oct 2022
|align=left|Carson
"""

SCHEDULE = """
<h2>Full schedule:</h2>
<h2>September</h2>
<p><b>Sept. 20: London</b></p>
<p>Old Bout vs. Somebody, 8 rounds, heavyweights</p>
<p><b>Sept. 26: London (Paramount+)</b></p>
<p>Johnny Fisher vs. Michael Pirotton, 10 rounds, heavyweights</p>
<p>Craig Richards vs. Radivoje Kalajozic, 10 rounds, light heavyweights</p>
<p><b>Sept. 26: Fuji, Japan</b></p>
<p>Title fight: Thanongsak Simsri vs. Mark Vicelles, 12 rounds, for a title</p>
<p><b>Oct. 17: Carson, California</b></p>
<p>Title fight: Sebastian Fundora vs. Ermal Hadribeaj, 12 rounds, for a title</p>
<p>Caleb Tirado vs. TBA, 6 rounds, bantamweights</p>
"""


def test_schedule_starts_at_the_next_named_bout():
    bouts = boxing.parse_schedule(SCHEDULE, today=date(2026, 9, 25))
    assert [(b["date"], b["a"], b["b"], b["title"]) for b in bouts] == [
        ("2026-09-26", "Johnny Fisher", "Michael Pirotton", False),
        ("2026-09-26", "Craig Richards", "Radivoje Kalajozic", False),
        ("2026-09-26", "Thanongsak Simsri", "Mark Vicelles", True),
        ("2026-10-17", "Sebastian Fundora", "Ermal Hadribeaj", True),
    ]


def test_us_dates_and_plain_opponents_count():
    wikitext = """
==Professional boxing record==
{|class=wikitable
|-
|2
|{{yes2}}Win
|2–0
|style="text-align:left;"|[[Tevin Farmer]]
|KO
|1 (10)
|[[Some fight|Jun 28, 2025]]
|Arena
|-
|1
|{{yes2}}Win
|1–0
|style="text-align:left;"|Rene Tellez Giron
|UD
|12
|Nov 2, 2024
|Arena
"""
    bouts = boxing.parse_bouts(wikitext, "Floyd Schofield")
    assert [(b["date"], b["opponent"]) for b in bouts] == [
        ("2024-11-02", "Rene Tellez Giron"),
        ("2025-06-28", "Tevin Farmer"),
    ]


def test_wikipedia_record_keeps_wins_losses_and_draws():
    bouts = boxing.parse_bouts(FUNDORA, "Sebastian Fundora")
    assert [(b["date"], b["result"], b["method"], b["opponent"]) for b in bouts] == [
        ("2022-10-08", "D", "SD", "Carlos Ocampo"),
        ("2023-04-08", "L", "KO", "Brian Mendoza"),
        ("2025-07-19", "W", "TKO", "Tim Tszyu"),
    ]
    assert bouts[0]["fighter_id"] == boxing.canon("Sebastian Fundora")


def test_shared_history_picks_the_fighter_who_beat_the_better_opponent():
    pages = {
        "Alpha": _record("Alpha", [("Win", "Beta", "KO", "1 Jan 2024"), ("Win", "Gamma", "UD", "1 Jun 2024")]),
        "Beta": _record("Beta", [("Loss", "Alpha", "KO", "1 Jan 2024"), ("Win", "Gamma", "SD", "1 Mar 2024")]),
        "Gamma": _record("Gamma", [("Loss", "Alpha", "UD", "1 Jun 2024"), ("Loss", "Beta", "SD", "1 Mar 2024")]),
    }
    schedule = [{"date": "2026-09-26", "place": "London", "title": False, "a": "Alpha", "b": "Gamma", "note": ""}]
    resolved = {"Alpha": "Alpha", "Beta": "Beta", "Gamma": "Gamma"}
    report = boxing.build_boxing_report(schedule, pages, resolved)
    assert report["next_bout"]["winner"]["name"] == "Alpha"
    assert report["next_bout"]["basis"] == "shared history"
    assert report["call"] == "Alpha wins."
    assert report["games_played"] == 3


def test_unconnected_records_still_name_a_winner():
    pages = {
        "Alpha": _record("Alpha", [("Win", "Nobody", "UD", "1 Jan 2025"), ("Win", "Nobody Two", "UD", "1 Jun 2025")]),
    }
    schedule = [{"date": "2026-09-26", "place": "Miami", "title": False, "a": "Alpha", "b": "Debut", "note": ""}]
    resolved = {"Alpha": "Alpha", "Debut": ""}
    report = boxing.build_boxing_report(schedule, pages, resolved)
    assert report["next_bout"]["winner"]["name"] == "Alpha"
    assert report["next_bout"]["basis"] == "own records"
    assert report["history"]["left"][0]["opponent"] == "Nobody"


def test_boxing_route(monkeypatch):
    import serve

    sample = {"call": "Alpha wins.", "detail": "from bouts already fought", "next_bout": None, "upcoming": []}
    monkeypatch.setattr(boxing, "get_boxing_report", lambda refresh=False: sample)
    serve.app.config.update(TESTING=True)
    client = serve.app.test_client()
    page = client.get("/football")
    assert page.status_code == 200
    assert b"Difference Engine" in page.data
    assert b"Boxing" in page.data
    assert client.get("/boxing").status_code == 200
    api = client.get("/api/boxing")
    assert api.status_code == 200
    assert api.get_json()["call"] == "Alpha wins."


def _record(fighter, rows):
    lines = ["==Professional boxing record==", "{|class=wikitable"]
    for index, (result, opponent, method, when) in enumerate(rows, start=1):
        mark = {"Win": "{{yes2}}Win", "Loss": "{{no2}}Loss", "Draw": "{{draw}}Draw"}[result]
        lines.append("|-")
        lines.append(f"|{index}")
        lines.append(f"|{mark}")
        lines.append("|1-0")
        lines.append(f"|align=left|[[{opponent}]]")
        lines.append(f"|{method}")
        lines.append("|12")
        lines.append(f"|{when}")
        lines.append("|Arena")
    return "\n".join(lines) + "\n"
