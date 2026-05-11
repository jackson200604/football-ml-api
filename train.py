import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import requests
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from dotenv import load_dotenv

load_dotenv()

FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_KEY")

# ============================================================
# 1. COLLECTE DES DONNÉES VIA FOOTBALL-DATA.ORG
# ============================================================

def fetch_matches(competition_code: str, season: int):
    """Récupère tous les matchs terminés d'une compétition"""
    headers = {"X-Auth-Token": FOOTBALL_DATA_KEY}
    url = f"https://api.football-data.org/v4/competitions/{competition_code}/matches"
    params = {"season": season, "status": "FINISHED"}
    r = requests.get(url, headers=headers, params=params)
    if r.status_code != 200:
        print(f"❌ Erreur {competition_code} {season}: {r.text}")
        return []
    return r.json().get("matches", [])

def fetch_all_matches():
    """Collecte les matchs de plusieurs ligues et saisons"""
    competitions = ["PL", "PD", "BL1", "FL1", "SA"]
    seasons      = [2021, 2022, 2023]
    all_matches  = []

    for comp in competitions:
        for season in seasons:
            print(f"📥 Récupération {comp} saison {season}...")
            matches = fetch_matches(comp, season)
            all_matches.extend(matches)
            print(f"   ✅ {len(matches)} matchs récupérés")

    print(f"\n📊 Total : {len(all_matches)} matchs collectés")
    return all_matches

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def compute_team_stats(matches, team_id, before_date, home_away="all", last_n=10):
    """
    Calcule les stats d'une équipe sur ses N derniers matchs
    avant une date donnée
    """
    team_matches = []

    for m in matches:
        if m["utcDate"] >= before_date:
            continue
        if m["score"]["fullTime"]["home"] is None:
            continue

        is_home = m["homeTeam"]["id"] == team_id
        is_away = m["awayTeam"]["id"] == team_id

        if not is_home and not is_away:
            continue
        if home_away == "home" and not is_home:
            continue
        if home_away == "away" and not is_away:
            continue

        if is_home:
            gf = m["score"]["fullTime"]["home"]
            ga = m["score"]["fullTime"]["away"]
            result = m["score"]["winner"]
            pts = 3 if result == "HOME_TEAM" else (1 if result == "DRAW" else 0)
        else:
            gf = m["score"]["fullTime"]["away"]
            ga = m["score"]["fullTime"]["home"]
            result = m["score"]["winner"]
            pts = 3 if result == "AWAY_TEAM" else (1 if result == "DRAW" else 0)

        team_matches.append({
            "date": m["utcDate"],
            "gf": gf,
            "ga": ga,
            "pts": pts
        })

    # Trier par date décroissante et prendre les N derniers
    team_matches = sorted(team_matches, key=lambda x: x["date"], reverse=True)[:last_n]

    if not team_matches:
        return {
            "goals_avg": 1.2,
            "conceded_avg": 1.2,
            "form": 5.0,
        }

    return {
        "goals_avg":    round(np.mean([m["gf"] for m in team_matches]), 3),
        "conceded_avg": round(np.mean([m["ga"] for m in team_matches]), 3),
        "form":         round(sum([m["pts"] for m in team_matches[:5]]), 1),
    }

def compute_h2h(matches, home_id, away_id, before_date, last_n=5):
    """Calcule les stats head-to-head entre deux équipes"""
    h2h = []

    for m in matches:
        if m["utcDate"] >= before_date:
            continue
        if m["score"]["fullTime"]["home"] is None:
            continue

        same_teams = (
            (m["homeTeam"]["id"] == home_id and m["awayTeam"]["id"] == away_id) or
            (m["homeTeam"]["id"] == away_id and m["awayTeam"]["id"] == home_id)
        )
        if not same_teams:
            continue

        winner = m["score"]["winner"]
        if m["homeTeam"]["id"] == home_id:
            if winner == "HOME_TEAM":   result = "home"
            elif winner == "DRAW":      result = "draw"
            else:                       result = "away"
        else:
            if winner == "AWAY_TEAM":   result = "home"
            elif winner == "DRAW":      result = "draw"
            else:                       result = "away"

        h2h.append(result)

    h2h = h2h[:last_n]
    return {
        "h2h_home_wins": h2h.count("home"),
        "h2h_draws":     h2h.count("draw"),
        "h2h_away_wins": h2h.count("away"),
    }

def build_dataset(matches):
    """Construit le dataset final avec toutes les features"""
    rows = []

    for m in matches:
        if m["score"]["fullTime"]["home"] is None:
            continue

        home_id    = m["homeTeam"]["id"]
        away_id    = m["awayTeam"]["id"]
        match_date = m["utcDate"]
        winner     = m["score"]["winner"]

        if winner == "HOME_TEAM":   label = 2
        elif winner == "DRAW":      label = 1
        else:                       label = 0

        # Stats domicile et extérieur
        home_stats = compute_team_stats(matches, home_id, match_date, "home")
        away_stats = compute_team_stats(matches, away_id, match_date, "away")
        h2h        = compute_h2h(matches, home_id, away_id, match_date)

        row = {
            "home_goals_avg":            home_stats["goals_avg"],
            "away_goals_avg":            away_stats["goals_avg"],
            "home_conceded_avg":         home_stats["conceded_avg"],
            "away_conceded_avg":         away_stats["conceded_avg"],
            "home_form":                 home_stats["form"],
            "away_form":                 away_stats["form"],
            "home_shots_avg":            home_stats["goals_avg"] * 5.5,
            "away_shots_avg":            away_stats["goals_avg"] * 5.5,
            "home_shots_on_target_avg":  home_stats["goals_avg"] * 3.2,
            "away_shots_on_target_avg":  away_stats["goals_avg"] * 3.2,
            "home_xg_avg":               home_stats["goals_avg"] * 0.95,
            "away_xg_avg":               away_stats["goals_avg"] * 0.95,
            "h2h_home_wins":             h2h["h2h_home_wins"],
            "h2h_draws":                 h2h["h2h_draws"],
            "h2h_away_wins":             h2h["h2h_away_wins"],
            "home_is_top6":              0,
            "away_is_top6":              0,
            "home_position":             10,
            "away_position":             10,
            "goal_diff_avg":             home_stats["goals_avg"] - away_stats["goals_avg"],
            "form_diff":                 home_stats["form"] - away_stats["form"],
            "xg_diff":                   home_stats["goals_avg"] * 0.95 - away_stats["goals_avg"] * 0.95,
            "shots_on_target_diff":      home_stats["goals_avg"] * 3.2 - away_stats["goals_avg"] * 3.2,
            "result":                    label,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"✅ Dataset construit : {len(df)} lignes, {len(df.columns)} colonnes")
    return df

# ============================================================
# 3. ENTRAÎNEMENT XGBOOST
# ============================================================

FEATURES = [
    "home_goals_avg", "away_goals_avg",
    "home_conceded_avg", "away_conceded_avg",
    "home_form", "away_form",
    "home_shots_avg", "away_shots_avg",
    "home_shots_on_target_avg", "away_shots_on_target_avg",
    "home_xg_avg", "away_xg_avg",
    "h2h_home_wins", "h2h_draws", "h2h_away_wins",
    "home_is_top6", "away_is_top6",
    "home_position", "away_position",
    "goal_diff_avg", "form_diff", "xg_diff",
    "shots_on_target_diff",
]

def train_model(df: pd.DataFrame):
    """Entraîne le modèle XGBoost et sauvegarde model.pkl"""
    X = df[FEATURES]
    y = df["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        eval_metric="mlogloss",
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    # Évaluation
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"\n🎯 Accuracy : {acc:.2%}")
    print("\n📊 Rapport détaillé :")
    print(classification_report(
        y_test, preds,
        target_names=["Away Win", "Draw", "Home Win"]
    ))

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"\n🔁 Cross-validation (5 folds) : {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

    # Sauvegarde
    joblib.dump(model, "model.pkl")
    print("\n✅ Modèle sauvegardé : model.pkl")

    return model

# ============================================================
# 4. PIPELINE PRINCIPAL
# ============================================================

if __name__ == "__main__":
    print("🚀 Démarrage de l'entraînement...\n")

    print("📥 Collecte des données...")
    matches = fetch_all_matches()

    print("\n⚙️  Construction du dataset...")
    df = build_dataset(matches)

    print("\n🤖 Entraînement XGBoost...")
    model = train_model(df)

    print("\n🏁 Terminé ! Le modèle est prêt.")
