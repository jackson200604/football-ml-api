from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os

router = APIRouter()

# ============================================================
# SCHÉMA DES DONNÉES D'ENTRÉE
# ============================================================

class MatchInput(BaseModel):
    # Stats offensives
    home_goals_avg: float        # Moyenne buts marqués à domicile
    away_goals_avg: float        # Moyenne buts marqués à l'extérieur
    home_conceded_avg: float     # Moyenne buts encaissés à domicile
    away_conceded_avg: float     # Moyenne buts encaissés à l'extérieur

    # Forme récente (points sur 5 derniers matchs, max=15)
    home_form: float
    away_form: float

    # Tirs
    home_shots_avg: float
    away_shots_avg: float
    home_shots_on_target_avg: float
    away_shots_on_target_avg: float

    # xG (expected goals)
    home_xg_avg: float
    away_xg_avg: float

    # Head-to-head (sur les 5 dernières confrontations)
    h2h_home_wins: int
    h2h_draws: int
    h2h_away_wins: int

    # Contexte
    home_is_top6: int            # 1 si top 6 du classement
    away_is_top6: int
    home_position: int           # Position au classement
    away_position: int

# ============================================================
# FEATURES UTILISÉES PAR LE MODÈLE
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
    # Features dérivées calculées automatiquement
    "goal_diff_avg",
    "form_diff",
    "xg_diff",
    "shots_on_target_diff",
]

# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================

def load_model():
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

# ============================================================
# ENDPOINTS
# ============================================================

@router.post("/match")
def predict_match(match: MatchInput):
    """
    Prédit le résultat d'un match
    Retourne les probabilités : victoire domicile, nul, victoire extérieur
    """
    model = load_model()

    # Calcul des features dérivées
    data = match.dict()
    data["goal_diff_avg"]          = data["home_goals_avg"] - data["away_goals_avg"]
    data["form_diff"]              = data["home_form"] - data["away_form"]
    data["xg_diff"]                = data["home_xg_avg"] - data["away_xg_avg"]
    data["shots_on_target_diff"]   = data["home_shots_on_target_avg"] - data["away_shots_on_target_avg"]

    df = pd.DataFrame([data])[FEATURES]

    # Si pas de modèle entraîné → prédiction de base par règles
    if model is None:
        return fallback_prediction(data)

    proba = model.predict_proba(df)[0]
    return build_response(proba, data)

def build_response(proba, data):
    home_win = round(float(proba[2]), 3)
    draw     = round(float(proba[1]), 3)
    away_win = round(float(proba[0]), 3)

    # Déterminer le favori
    if home_win > draw and home_win > away_win:
        favorite = "Domicile"
        confidence = home_win
    elif away_win > draw and away_win > home_win:
        favorite = "Extérieur"
        confidence = away_win
    else:
        favorite = "Match nul probable"
        confidence = draw

    return {
        "probabilities": {
            "home_win": home_win,
            "draw":     draw,
            "away_win": away_win,
        },
        "prediction": favorite,
        "confidence": f"{round(confidence * 100, 1)}%",
        "model_used": "XGBoost",
    }

def fallback_prediction(data):
    """
    Prédiction basique par règles si le modèle n'est pas encore entraîné.
    Basé sur la forme et les moyennes de buts.
    """
    home_score = (
        data["home_goals_avg"] * 0.3 +
        data["home_form"] / 15 * 0.3 +
        data["home_xg_avg"] * 0.2 +
        (1 - data["home_conceded_avg"] / 5) * 0.2
    )
    away_score = (
        data["away_goals_avg"] * 0.3 +
        data["away_form"] / 15 * 0.3 +
        data["away_xg_avg"] * 0.2 +
        (1 - data["away_conceded_avg"] / 5) * 0.2
    )

    total = home_score + away_score + 0.3
    home_win = round(home_score / total, 3)
    away_win = round(away_score / total, 3)
    draw     = round(1 - home_win - away_win, 3)

    if home_win > draw and home_win > away_win:
        favorite = "Domicile"
        confidence = home_win
    elif away_win > draw:
        favorite = "Extérieur"
        confidence = away_win
    else:
        favorite = "Match nul probable"
        confidence = draw

    return {
        "probabilities": {
            "home_win": home_win,
            "draw":     draw,
            "away_win": away_win,
        },
        "prediction": favorite,
        "confidence": f"{round(confidence * 100, 1)}%",
        "model_used": "Règles (modèle non encore entraîné)",
    }

@router.get("/status")
def model_status():
    """Vérifie si le modèle XGBoost est entraîné et disponible"""
    model = load_model()
    if model is None:
        return {
            "status": "⚠️ Modèle non entraîné",
            "message": "Lance POST /train pour entraîner le modèle",
            "fallback": "Prédiction par règles active"
        }
    return {
        "status": "✅ Modèle XGBoost prêt",
        "features": FEATURES,
        "classes": ["away_win", "draw", "home_win"]
    }
