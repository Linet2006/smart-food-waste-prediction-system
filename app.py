"""
================================================
 SMART FOOD WASTE PREDICTION SYSTEM
 app.py  —  Run this AFTER train_model.py
 LearnDepth Academy LLP | Internship Project
================================================
 HOW TO RUN:
   python app.py
 THEN OPEN:
   http://127.0.0.1:5000
================================================
"""

import warnings
warnings.filterwarnings("ignore")

import pickle, json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Load saved model ─────────────────────────────────────
with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)

MODEL        = bundle["model"]
FEATURE_COLS = bundle["feature_cols"]
MODEL_NAME   = bundle["model_name"]
METRICS      = bundle["metrics"]
ALL_RESULTS  = bundle["all_results"]

with open("static/metrics.json") as f:
    METRICS_JSON = json.load(f)

# ── Constants ─────────────────────────────────────────────
DAY_ORDER   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
WEATHER_MAP = {"Sunny":3,"Cloudy":2,"Rainy":1,"Stormy":0}
COST_PER_MEAL = 5.0
CO2_PER_MEAL  = 2.5


def make_features(day, weather, festival, expected, prev_day, prev_week):
    day_num    = DAY_ORDER.index(day)
    weather_n  = WEATHER_MAP[weather]
    is_weekend = 1 if day_num >= 5 else 0
    is_friday  = 1 if day_num == 4 else 0
    demand_lag = (prev_day + prev_week) / 2.0
    cust_wx    = expected * weather_n / 3.0
    fest_cust  = festival * expected

    row = {
        "Expected_Customers":       expected,
        "Previous_Day_Consumption": prev_day,
        "Previous_Week_Same_Day":   prev_week,
        "Festival":                 festival,
        "Is_Weekend":               is_weekend,
        "Is_Friday":                is_friday,
        "Demand_Avg_Lag":           demand_lag,
        "Customer_Weather_Score":   cust_wx,
        "Festival_X_Customers":     fest_cust,
        "Weather_Num":              weather_n,
        "Day_Num":                  day_num,
    }
    return pd.DataFrame([row])[FEATURE_COLS]


def predict_full(features):
    pred  = float(MODEL.predict(features)[0])
    trees = np.array([t.predict(features.values)[0] for t in MODEL.estimators_])
    std   = float(trees.std())
    lower = max(0, pred - 1.5 * std)
    upper = pred + 1.5 * std
    return round(pred), round(lower), round(upper), round(std, 1)


# ── Routes ────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
        model_name = MODEL_NAME,
        mae  = round(METRICS["MAE"],  2),
        rmse = round(METRICS["RMSE"], 2),
        r2   = round(METRICS["R2"],   4),
        days     = DAY_ORDER,
        weathers = list(WEATHER_MAP.keys()),
    )


@app.route("/predict", methods=["POST"])
def predict():
    d        = request.get_json()
    day      = d["day"]
    weather  = d["weather"]
    festival = int(d["festival"])
    expected = int(d["expected"])
    prev_day = int(d["prev_day"])
    prev_wk  = int(d["prev_week"])

    feats = make_features(day, weather, festival, expected, prev_day, prev_wk)
    pred, lower, upper, std = predict_full(feats)

    # Smart buffer
    if festival:
        buffer = 0.08
    elif weather == "Sunny":
        buffer = 0.05
    elif weather in ["Rainy","Stormy"]:
        buffer = 0.02
    else:
        buffer = 0.04
    recommended = round(pred * (1 + buffer))

    # Impact
    over_prep   = max(0, expected - pred)
    cost_saved  = round(over_prep * COST_PER_MEAL, 2)
    co2_saved   = round(over_prep * CO2_PER_MEAL,  1)
    efficiency  = max(0, min(100, round(100 - (std / max(pred,1)) * 100)))

    # Feature contributions
    imps  = MODEL.feature_importances_
    vals  = feats.values[0]
    contribs = sorted(
        [(col, round(float(imp * abs(v)), 2)) for col, imp, v in zip(FEATURE_COLS, imps, vals)],
        key=lambda x: x[1], reverse=True
    )[:3]

    return jsonify({
        "prediction":  pred,
        "lower":       lower,
        "upper":       upper,
        "recommended": recommended,
        "buffer_pct":  round(buffer * 100, 1),
        "cost_saved":  cost_saved,
        "co2_saved":   co2_saved,
        "efficiency":  efficiency,
        "top_features":contribs,
        "std":         std,
    })


@app.route("/weekly", methods=["POST"])
def weekly():
    d        = request.get_json()
    base_exp = int(d["base_expected"])
    base_pre = int(d["base_prev"])
    wx_list  = d.get("weather_list", ["Sunny"]*7)

    plan = []
    prev = base_pre
    for i, day in enumerate(DAY_ORDER):
        weather  = wx_list[i] if i < len(wx_list) else "Sunny"
        festival = 1 if day in ["Friday","Saturday"] else 0
        expected = base_exp + int(np.random.randint(-20, 30))
        feats    = make_features(day, weather, festival, expected, prev, base_pre)
        pred, lo, hi, _ = predict_full(feats)
        recommended = round(pred * 1.06)
        plan.append({
            "day": day, "weather": weather, "festival": bool(festival),
            "expected": int(expected),
            "predicted": pred, "lower": lo, "upper": hi,
            "recommended": recommended,
            "cost": round(recommended * COST_PER_MEAL * 0.2),
        })
        prev = pred

    return jsonify({
        "plan":        plan,
        "total_meals": sum(p["recommended"] for p in plan),
        "total_cost":  sum(p["cost"] for p in plan),
    })


@app.route("/dashboard_data")
def dashboard_data():
    return jsonify({
        "model_name": MODEL_NAME,
        "metrics":    METRICS_JSON,
        "plots": {
            "distribution": "/static/plots/eda_distribution.png",
            "day_impact":   "/static/plots/eda_day_impact.png",
            "weather":      "/static/plots/eda_weather_festival.png",
            "heatmap":      "/static/plots/eda_heatmap.png",
            "comparison":   "/static/plots/model_comparison.png",
            "actual_pred":  "/static/plots/actual_vs_predicted.png",
            "importance":   "/static/plots/feature_importance.png",
            "residuals":    "/static/plots/residuals.png",
        }
    })


if __name__ == "__main__":
    print("\n  Smart Food Waste Prediction System")
    print("  Open: http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
