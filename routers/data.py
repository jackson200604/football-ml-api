from fastapi import APIRouter, HTTPException
import requests
import os

router = APIRouter()

FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_KEY")
OPENLIGADB_BASE   = "https://api.openligadb.de"
ESPN_BASE         = "https://site.api.espn.com/apis/site/v2/sports/soccer"
SOFASCORE_BASE    = "https://api.sofascore.com/api/v1"

HEADERS_SOFASCORE = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.sofascore.com/"
}

# ============================================================
# 1. FOOTBALL-DATA.ORG (officielle — ta clé)
# ============================================================

@router.get("/competitions")
def get_competitions():
    """Liste des compétitions disponibles"""
    headers = {"X-Auth-Token": FOOTBALL_DATA_KEY}
    r = requests.get("https://api.football-data.org/v4/competitions", headers=headers)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

@router.get("/matches/{competition_code}")
def get_matches(competition_code: str, matchday: int = None):
    """
    Matchs d'une compétition
    Codes : PL=Premier League, PD=La Liga,
            BL1=Bundesliga, FL1=Ligue 1, SA=Serie A
    """
    headers = {"X-Auth-Token": FOOTBALL_DATA_KEY}
    url = f"https://api.football-data.org/v4/competitions/{competition_code}/matches"
    params = {}
    if matchday:
        params["matchday"] = matchday
    r = requests.get(url, headers=headers, params=params)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

@router.get("/team/{team_id}/matches")
def get_team_matches(team_id: int, limit: int = 10):
    """Derniers matchs d'une équipe"""
    headers = {"X-Auth-Token": FOOTBALL_DATA_KEY}
    url = f"https://api.football-data.org/v4/teams/{team_id}/matches"
    params = {"limit": limit, "status": "FINISHED"}
    r = requests.get(url, headers=headers, params=params)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

@router.get("/standings/{competition_code}")
def get_standings(competition_code: str):
    """Classement d'une compétition"""
    headers = {"X-Auth-Token": FOOTBALL_DATA_KEY}
    url = f"https://api.football-data.org/v4/competitions/{competition_code}/standings"
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

@router.get("/h2h/{match_id}")
def get_h2h(match_id: int):
    """Head-to-head entre deux équipes"""
    headers = {"X-Auth-Token": FOOTBALL_DATA_KEY}
    url = f"https://api.football-data.org/v4/matches/{match_id}/head2head"
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

# ============================================================
# 2. ESPN API NON-OFFICIELLE (sans clé)
# ============================================================

@router.get("/espn/leagues")
def get_espn_leagues():
    """
    Ligues supportées par ESPN :
    eng.1=Premier League, esp.1=La Liga,
    ger.1=Bundesliga, fra.1=Ligue 1,
    ita.1=Serie A, usa.1=MLS
    """
    return {
        "leagues": [
            {"name": "Premier League",  "code": "eng.1"},
            {"name": "La Liga",         "code": "esp.1"},
            {"name": "Bundesliga",      "code": "ger.1"},
            {"name": "Ligue 1",         "code": "fra.1"},
            {"name": "Serie A",         "code": "ita.1"},
            {"name": "Champions League","code": "uefa.champions"},
            {"name": "MLS",             "code": "usa.1"},
        ]
    }

@router.get("/espn/scoreboard/{league}")
def get_espn_scoreboard(league: str):
    """
    Scores en direct / récents ESPN
    Exemple : /espn/scoreboard/eng.1
    """
    url = f"{ESPN_BASE}/{league}/scoreboard"
    r = requests.get(url)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

@router.get("/espn/standings/{league}")
def get_espn_standings(league: str):
    """Classement ESPN — Exemple : /espn/standings/eng.1"""
    url = f"{ESPN_BASE}/{league}/standings"
    r = requests.get(url)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

@router.get("/espn/team/{league}/{team_id}/matches")
def get_espn_team_matches(league: str, team_id: int):
    """Matchs d'une équipe via ESPN"""
    url = f"{ESPN_BASE}/{league}/teams/{team_id}/schedule"
    r = requests.get(url)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

@router.get("/espn/match/{league}/{match_id}/stats")
def get_espn_match_stats(league: str, match_id: str):
    """Stats détaillées d'un match ESPN (possession, tirs, etc.)"""
    url = f"{ESPN_BASE}/{league}/summary"
    params = {"event": match_id}
    r = requests.get(url, params=params)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

@router.get("/espn/news/{league}")
def get_espn_news(league: str):
    """Actualités football ESPN"""
    url = f"{ESPN_BASE}/{league}/news"
    r = requests.get(url)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

# ============================================================
# 3. SOFASCORE NON-OFFICIELLE (sans clé)
# ============================================================

@router.get("/sofascore/match/{match_id}/stats")
def get_sofascore_match_stats(match_id: int):
    """
    Stats complètes d'un match SofaScore
    (possession, xG, tirs cadrés, duels, etc.)
    """
    url = f"{SOFASCORE_BASE}/event/{match_id}/statistics"
    r = requests.get(url, headers=HEADERS_SOFASCORE)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

@router.get("/sofascore/match/{match_id}/lineups")
def get_sofascore_lineups(match_id: int):
    """Compositions d'équipes via SofaScore"""
    url = f"{SOFASCORE_BASE}/event/{match_id}/lineups"
    r = requests.get(url, headers=HEADERS_SOFASCORE)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

@router.get("/sofascore/team/{team_id}/stats/{tournament_id}/{season_id}")
def get_sofascore_team_stats(team_id: int, tournament_id: int, season_id: int):
    """
    Stats saison d'une équipe SofaScore
    (buts, xG, clean sheets, forme, etc.)
    """
    url = f"{SOFASCORE_BASE}/team/{team_id}/tournament/{tournament_id}/season/{season_id}/statistics/overall"
    r = requests.get(url, headers=HEADERS_SOFASCORE)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

@router.get("/sofascore/team/{team_id}/form")
def get_sofascore_team_form(team_id: int):
    """Forme récente d'une équipe (5 derniers matchs)"""
    url = f"{SOFASCORE_BASE}/team/{team_id}/events/last/0"
    r = requests.get(url, headers=HEADERS_SOFASCORE)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

@router.get("/sofascore/live")
def get_sofascore_live():
    """Tous les matchs en direct sur SofaScore"""
    url = f"{SOFASCORE_BASE}/sport/football/events/live"
    r = requests.get(url, headers=HEADERS_SOFASCORE)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

# ============================================================
# 4. OPENLIGADB (sans clé — Bundesliga)
# ============================================================

@router.get("/bundesliga/matchday/{season}/{matchday}")
def get_bundesliga_matchday(season: int, matchday: int):
    """Matchs Bundesliga — Exemple : /bundesliga/matchday/2023/10"""
    url = f"{OPENLIGADB_BASE}/getmatchdata/bl1/{season}/{matchday}"
    r = requests.get(url)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

@router.get("/bundesliga/table/{season}")
def get_bundesliga_table(season: int):
    """Classement Bundesliga"""
    url = f"{OPENLIGADB_BASE}/getbltable/bl1/{season}"
    r = requests.get(url)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

@router.get("/bundesliga/team/{team_id}/matches/{season}")
def get_bundesliga_team_matches(team_id: int, season: int):
    """Tous les matchs d'une équipe Bundesliga sur une saison"""
    url = f"{OPENLIGADB_BASE}/getmatchdata/bl1/{season}"
    r = requests.get(url)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    matches = r.json()
    team_matches = [
        m for m in matches
        if m["team1"]["teamId"] == team_id or m["team2"]["teamId"] == team_id
    ]
    return team_matches
