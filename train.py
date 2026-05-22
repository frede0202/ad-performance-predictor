"""
train.py
Trains and evaluates a ROAS prediction model using XGBoost.
Outputs feature importances and model metrics.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/ad_campaigns.csv"
OUTPUT_DIR  = "outputs"
RANDOM_SEED = 42

CATEGORICAL_COLS = ["platform", "ad_format", "objective", "audience_size"]
FEATURE_COLS = [
    "platform", "ad_format", "objective", "audience_size",
    "daily_budget", "creative_quality", "days_running",
    "retargeting", "day_of_week", "ctr", "cpc",
]
TARGET_COL = "roas"


def load_and_preprocess(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y, encoders


def train_model(X_train, y_train):
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    print(f"\n── Model Performance ──────────────────")
    print(f"  MAE : {mae:.4f}")
    print(f"  R²  : {r2:.4f}")
    print(f"───────────────────────────────────────\n")
    return preds, {"mae": round(mae, 4), "r2": round(r2, 4)}


def plot_feature_importance(model, feature_names: list, output_dir: str):
    importance = model.feature_importances_
    fi = pd.Series(importance, index=feature_names).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(fi.index, fi.values, color="#4F6AF5", edgecolor="white")
    ax.set_xlabel("Feature Importance (gain)", fontsize=11)
    ax.set_title("What drives ROAS?", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    plt.tight_layout()

    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    return fi


def plot_predictions(y_test, preds, output_dir: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, preds, alpha=0.3, s=15, color="#4F6AF5")
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual ROAS")
    ax.set_ylabel("Predicted ROAS")
    ax.set_title("Predicted vs Actual ROAS", fontsize=13, fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = os.path.join(output_dir, "predicted_vs_actual.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")


def top_insights(fi: pd.Series):
    print("── Top Insights ────────────────────────")
    for feat, score in fi.sort_values(ascending=False).head(5).items():
        print(f"  {feat:<20} importance: {score:.3f}")
    print("────────────────────────────────────────\n")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    X, y, _ = load_and_preprocess(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    print("Training XGBoost model...")
    model = train_model(X_train, y_train)

    preds, metrics = evaluate(model, X_test, y_test)

    fi = plot_feature_importance(model, FEATURE_COLS, OUTPUT_DIR)
    plot_predictions(y_test, preds, OUTPUT_DIR)
    top_insights(fi)

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Done! Check the outputs/ folder.")
