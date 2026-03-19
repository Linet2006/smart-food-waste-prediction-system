"""
================================================
 SMART FOOD WASTE PREDICTION SYSTEM
 train_model.py  —  Run this FIRST
 LearnDepth Academy LLP | Internship Project
================================================
 HOW TO RUN:
   python train_model.py
 WHAT IT DOES:
   - Loads dataset.csv
   - Preprocesses & engineers features
   - Trains 3 ML models & picks the best
   - Saves model.pkl + all EDA/eval plots
================================================
"""

import os, warnings, json, pickle
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Folders ──────────────────────────────────────────────
os.makedirs("static/plots", exist_ok=True)

# ── Constants ─────────────────────────────────────────────
DAY_ORDER   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
WEATHER_MAP = {"Sunny":3,"Cloudy":2,"Rainy":1,"Stormy":0}

# ═══════════════════════════════════════════════════════════
# STEP 1 — DATA LOADING
# ═══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  STEP 1 — DATA LOADING")
print("="*55)

df = pd.read_csv("dataset.csv")
print(f"  Rows   : {df.shape[0]}")
print(f"  Columns: {df.shape[1]}")
print(f"  Names  : {list(df.columns)}")
print(df.head(3).to_string())

# ═══════════════════════════════════════════════════════════
# STEP 2 — PREPROCESSING
# ═══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  STEP 2 — PREPROCESSING")
print("="*55)
print(f"  Missing values:\n{df.isnull().sum().to_string()}")

df["Day_Num"]     = df["Day_of_Week"].map({d: i for i, d in enumerate(DAY_ORDER)})
df["Weather_Num"] = df["Weather"].map(WEATHER_MAP)
print("  Encoding done.")

# ═══════════════════════════════════════════════════════════
# STEP 3 — EDA PLOTS
# ═══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  STEP 3 — EDA")
print("="*55)

BG   = "#0A0F1E"
CARD = "#111827"
CLR  = ["#00F5A0","#00D9F5","#7C3AED","#F59E0B","#EC4899","#EF4444","#10B981"]

def dark_fig(w=14, h=5, cols=2):
    fig, axes = plt.subplots(1, cols, figsize=(w, h))
    fig.patch.set_facecolor(BG)
    if cols == 1: axes = [axes]
    for ax in axes:
        ax.set_facecolor(CARD)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values(): sp.set_edgecolor("#1F2937")
    return fig, axes

# Plot 1 — Distribution & Scatter
fig, axes = dark_fig(14, 5, 2)
axes[0].hist(df["Meals_Consumed"], bins=30, color="#00F5A0", edgecolor="#0A0F1E", alpha=0.9)
axes[0].set_title("Distribution of Meals Consumed", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Meals Consumed"); axes[0].set_ylabel("Frequency")

axes[1].scatter(df["Expected_Customers"], df["Meals_Consumed"],
                alpha=0.4, c="#00D9F5", s=14)
axes[1].set_title("Expected Customers vs Meals Consumed", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Expected Customers"); axes[1].set_ylabel("Meals Consumed")
plt.tight_layout(pad=2)
plt.savefig("static/plots/eda_distribution.png", dpi=120, bbox_inches="tight", facecolor=BG)
plt.close()

# Plot 2 — Day of Week
day_avg = df.groupby("Day_of_Week")["Meals_Consumed"].mean().reindex(DAY_ORDER)
fig, axes = dark_fig(12, 5, 1)
bars = axes[0].bar(day_avg.index, day_avg.values, color=CLR, edgecolor="#0A0F1E", width=0.6)
axes[0].set_title("Average Meals by Day of Week", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Avg Meals Consumed")
for bar, val in zip(bars, day_avg.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                 f"{val:.0f}", ha="center", color="white", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig("static/plots/eda_day_impact.png", dpi=120, bbox_inches="tight", facecolor=BG)
plt.close()

# Plot 3 — Weather & Festival
fig, axes = dark_fig(14, 5, 2)
weather_order = ["Sunny","Cloudy","Rainy","Stormy"]
w_colors = ["#F59E0B","#94A3B8","#00D9F5","#7C3AED"]
weather_avg = df.groupby("Weather")["Meals_Consumed"].mean().reindex(weather_order)
axes[0].bar(weather_avg.index, weather_avg.values, color=w_colors, edgecolor="#0A0F1E")
axes[0].set_title("Avg Meals by Weather", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Avg Meals")
for bar, val in zip(axes[0].patches, weather_avg.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                 f"{val:.0f}", ha="center", color="white", fontsize=10, fontweight="bold")

fest_avg = df.groupby("Festival")["Meals_Consumed"].mean()
axes[1].bar(["Normal Day","Festival Day"], fest_avg.values,
            color=["#00F5A0","#EC4899"], edgecolor="#0A0F1E")
axes[1].set_title("Festival vs Normal Day", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Avg Meals")
for bar, val in zip(axes[1].patches, fest_avg.values):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                 f"{val:.0f}", ha="center", color="white", fontsize=10, fontweight="bold")
plt.tight_layout(pad=2)
plt.savefig("static/plots/eda_weather_festival.png", dpi=120, bbox_inches="tight", facecolor=BG)
plt.close()

# Plot 4 — Correlation Heatmap
corr_cols = ["Expected_Customers","Previous_Day_Consumption","Previous_Week_Same_Day",
             "Festival","Day_Num","Weather_Num","Meals_Consumed"]
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
            linewidths=0.5, linecolor=BG, annot_kws={"size":10,"color":"white"})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", color="white", pad=15)
ax.tick_params(colors="white", labelsize=9)
plt.tight_layout()
plt.savefig("static/plots/eda_heatmap.png", dpi=120, bbox_inches="tight", facecolor=BG)
plt.close()
print("  EDA plots saved ✓")

# ═══════════════════════════════════════════════════════════
# STEP 4 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  STEP 4 — FEATURE ENGINEERING")
print("="*55)

df["Is_Weekend"]              = (df["Day_Num"] >= 5).astype(int)
df["Is_Friday"]               = (df["Day_Num"] == 4).astype(int)
df["Demand_Avg_Lag"]          = (df["Previous_Day_Consumption"] + df["Previous_Week_Same_Day"]) / 2.0
df["Customer_Weather_Score"]  = df["Expected_Customers"] * df["Weather_Num"] / 3.0
df["Festival_X_Customers"]    = df["Festival"] * df["Expected_Customers"]

FEATURE_COLS = [
    "Expected_Customers", "Previous_Day_Consumption", "Previous_Week_Same_Day",
    "Festival", "Is_Weekend", "Is_Friday", "Demand_Avg_Lag",
    "Customer_Weather_Score", "Festival_X_Customers", "Weather_Num", "Day_Num"
]
TARGET = "Meals_Consumed"
X = df[FEATURE_COLS]
y = df[TARGET]
print(f"  Total features: {len(FEATURE_COLS)}")

# ═══════════════════════════════════════════════════════════
# STEP 5 — TRAIN / TEST SPLIT
# ═══════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n  Train: {len(X_train)} rows | Test: {len(X_test)} rows")

# ═══════════════════════════════════════════════════════════
# STEPS 6-7 — MODEL TRAINING
# ═══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  STEPS 6-7 — MODEL TRAINING")
print("="*55)

models = {
    "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=12,
                                               min_samples_leaf=2, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
                                                   max_depth=5, subsample=0.85, random_state=42),
    "Extra Trees":       ExtraTreesRegressor(n_estimators=200, max_depth=12,
                                             random_state=42, n_jobs=-1),
}

results = {}
trained  = {}
for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    preds = mdl.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)
    cv    = -cross_val_score(mdl, X, y, cv=5, scoring="neg_mean_absolute_error").mean()
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "CV_MAE": cv}
    trained[name] = mdl
    print(f"  {name:25s}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}  CV={cv:.2f}")

# ═══════════════════════════════════════════════════════════
# STEP 8 — EVALUATION
# ═══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  STEP 8 — EVALUATION")
print("="*55)

best_name  = min(results, key=lambda k: results[k]["MAE"])
best_model = trained[best_name]
best_preds = best_model.predict(X_test)
print(f"  Best Model : {best_name}")
print(f"  MAE        : {results[best_name]['MAE']:.2f}")
print(f"  RMSE       : {results[best_name]['RMSE']:.2f}")
print(f"  R²         : {results[best_name]['R2']:.4f}")

# Plot 5 — Model Comparison
fig, axes = dark_fig(15, 5, 3)
metric_list = ["MAE", "RMSE", "R2"]
colors_m    = ["#00F5A0","#00D9F5","#7C3AED"]
labels_m    = list(results.keys())
for ax, metric, color in zip(axes, metric_list, colors_m):
    vals = [results[m][metric] for m in labels_m]
    bars = ax.bar([l.replace(" ","\n") for l in labels_m], vals,
                  color=color, edgecolor="#0A0F1E", alpha=0.9, width=0.5)
    ax.set_title(metric, fontsize=14, fontweight="bold")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                f"{val:.3f}", ha="center", color="white", fontsize=9, fontweight="bold")
fig.suptitle("Model Comparison", fontsize=15, fontweight="bold", color="white")
plt.tight_layout()
plt.savefig("static/plots/model_comparison.png", dpi=120, bbox_inches="tight", facecolor=BG)
plt.close()

# Plot 6 — Actual vs Predicted
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
ax.scatter(y_test, best_preds, alpha=0.5, c="#00F5A0", s=16)
lims = [min(y_test.min(), best_preds.min())-10, max(y_test.max(), best_preds.max())+10]
ax.plot(lims, lims, "r--", linewidth=1.5, alpha=0.8, label="Perfect Prediction")
ax.set_xlabel("Actual", color="white"); ax.set_ylabel("Predicted", color="white")
ax.set_title(f"Actual vs Predicted — {best_name}", fontsize=13, fontweight="bold", color="white")
ax.tick_params(colors="white")
ax.legend(facecolor=CARD, labelcolor="white", edgecolor="#1F2937")
for sp in ax.spines.values(): sp.set_edgecolor("#1F2937")
plt.tight_layout()
plt.savefig("static/plots/actual_vs_predicted.png", dpi=120, bbox_inches="tight", facecolor=BG)
plt.close()

# Plot 7 — Feature Importance
fi = pd.Series(best_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
colors_fi = plt.cm.cool(np.linspace(0.3, 0.9, len(fi)))
bars = ax.barh(fi.index, fi.values, color=colors_fi, edgecolor="#0A0F1E")
ax.set_title(f"Feature Importance — {best_name}", fontsize=14, fontweight="bold", color="white")
ax.set_xlabel("Importance", color="white"); ax.tick_params(colors="white")
for sp in ax.spines.values(): sp.set_edgecolor("#1F2937")
for bar, val in zip(bars, fi.values):
    ax.text(val+0.002, bar.get_y()+bar.get_height()/2,
            f"{val:.3f}", va="center", color="white", fontsize=8.5, fontweight="bold")
plt.tight_layout()
plt.savefig("static/plots/feature_importance.png", dpi=120, bbox_inches="tight", facecolor=BG)
plt.close()

# Plot 8 — Residuals
residuals = y_test.values - best_preds
fig, axes = dark_fig(14, 5, 2)
axes[0].scatter(best_preds, residuals, alpha=0.4, c="#F59E0B", s=14)
axes[0].axhline(0, color="#EF4444", linestyle="--", linewidth=1.5)
axes[0].set_title("Residuals vs Predicted", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")
axes[1].hist(residuals, bins=30, color="#7C3AED", edgecolor="#0A0F1E", alpha=0.85)
axes[1].axvline(0, color="#EF4444", linestyle="--", linewidth=1.5)
axes[1].set_title("Residual Distribution", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Frequency")
plt.tight_layout(pad=2)
plt.savefig("static/plots/residuals.png", dpi=120, bbox_inches="tight", facecolor=BG)
plt.close()
print("  All evaluation plots saved ✓")

# ═══════════════════════════════════════════════════════════
# STEP 9 — SAVE MODEL
# ═══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  STEP 9 — SAVING MODEL")
print("="*55)

model_bundle = {
    "model":        best_model,
    "feature_cols": FEATURE_COLS,
    "model_name":   best_name,
    "metrics":      results[best_name],
    "all_results":  results,
}
with open("model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

# Save metrics JSON for the dashboard
metrics_json = {
    "best_model":  best_name,
    "mae":         round(results[best_name]["MAE"],  2),
    "rmse":        round(results[best_name]["RMSE"], 2),
    "r2":          round(results[best_name]["R2"],   4),
    "all_results": {k: {m: round(v,4) for m,v in v2.items()} for k,v2 in results.items()},
    "feature_importance": {k: round(float(v),4)
                           for k,v in zip(FEATURE_COLS, best_model.feature_importances_)},
}
with open("static/metrics.json", "w") as f:
    json.dump(metrics_json, f)

print(f"  model.pkl saved  ({best_name})")
print("\n" + "="*55)
print("  TRAINING COMPLETE — now run:  python app.py")
print("="*55 + "\n")
