"""
predict.py
Demonstrates how the trained model could be used to score new ad campaigns
before spending budget — the core use-case for Marketer.com.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from train import load_and_preprocess, train_model, FEATURE_COLS
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42


def build_model():
    """Retrain model on full dataset for prediction demo."""
    X, y, encoders = load_and_preprocess("data/ad_campaigns.csv")
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    model = train_model(X_train, y_train)
    return model, encoders


def encode_campaign(campaign: dict, encoders: dict) -> pd.DataFrame:
    df = pd.DataFrame([campaign])
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])
    return df[FEATURE_COLS]


def predict_roas(campaigns: list[dict]) -> pd.DataFrame:
    model, encoders = build_model()
    results = []
    for camp in campaigns:
        X = encode_campaign(camp, encoders)
        predicted_roas = model.predict(X)[0]
        results.append({**camp, "predicted_roas": round(float(predicted_roas), 2)})
    return pd.DataFrame(results).sort_values("predicted_roas", ascending=False)


if __name__ == "__main__":
    # Simulate 4 campaign variants a marketer might want to compare
    candidates = [
        {
            "platform": "Meta", "ad_format": "video", "objective": "conversions",
            "audience_size": "narrow", "daily_budget": 500, "creative_quality": 5,
            "days_running": 14, "retargeting": 1, "day_of_week": 1, "ctr": 0.04, "cpc": 1.2,
        },
        {
            "platform": "Google", "ad_format": "image", "objective": "traffic",
            "audience_size": "broad", "daily_budget": 300, "creative_quality": 3,
            "days_running": 7, "retargeting": 0, "day_of_week": 3, "ctr": 0.02, "cpc": 2.5,
        },
        {
            "platform": "Meta", "ad_format": "carousel", "objective": "conversions",
            "audience_size": "medium", "daily_budget": 800, "creative_quality": 4,
            "days_running": 30, "retargeting": 0, "day_of_week": 0, "ctr": 0.03, "cpc": 1.8,
        },
        {
            "platform": "Google", "ad_format": "video", "objective": "conversions",
            "audience_size": "narrow", "daily_budget": 1000, "creative_quality": 5,
            "days_running": 21, "retargeting": 1, "day_of_week": 2, "ctr": 0.05, "cpc": 1.0,
        },
    ]

    print("\n── Predicting ROAS for campaign variants ──────────────────────────")
    results = predict_roas(candidates)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(results[["platform", "ad_format", "objective", "audience_size",
                    "creative_quality", "retargeting", "predicted_roas"]].to_string(index=False))
    print("\n✅ Best predicted campaign is at the top.")
    print("💡 This is how Marketer could rank creatives before spending budget.\n")
