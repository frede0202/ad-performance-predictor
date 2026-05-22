"""
generate_data.py
Generates synthetic ad campaign data that mimics real Meta/Google Ads patterns.
"""

import numpy as np
import pandas as pd

def generate_ad_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    # --- Campaign features ---
    platforms        = np.random.choice(["Meta", "Google"], n_samples, p=[0.6, 0.4])
    ad_formats       = np.random.choice(["video", "image", "carousel"], n_samples, p=[0.4, 0.35, 0.25])
    objectives       = np.random.choice(["conversions", "traffic", "awareness"], n_samples, p=[0.5, 0.3, 0.2])
    audience_sizes   = np.random.choice(["narrow", "medium", "broad"], n_samples, p=[0.3, 0.4, 0.3])
    daily_budget     = np.random.uniform(50, 2000, n_samples)
    creative_quality = np.random.randint(1, 6, n_samples)          # 1–5 score
    days_running     = np.random.randint(1, 90, n_samples)
    retargeting      = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    day_of_week      = np.random.randint(0, 7, n_samples)          # 0=Mon

    # --- Derive CTR with realistic noise ---
    base_ctr = 0.02
    ctr = (
        base_ctr
        + (platforms == "Meta")         * 0.005
        + (ad_formats == "video")       * 0.008
        + (ad_formats == "carousel")    * 0.003
        + (objectives == "conversions") * 0.004
        + (audience_sizes == "narrow")  * 0.006
        + (audience_sizes == "broad")   * -0.003
        + (retargeting == 1)            * 0.010
        + (creative_quality - 3)        * 0.003
        + np.random.normal(0, 0.005, n_samples)
    ).clip(0.001, 0.15)

    # --- Derive CPC ---
    base_cpc = 1.5
    cpc = (
        base_cpc
        + (platforms == "Google")        * 0.8
        + (objectives == "conversions")  * 0.5
        + (audience_sizes == "narrow")   * 0.4
        + (daily_budget / 2000)          * 0.6
        + np.random.normal(0, 0.3, n_samples)
    ).clip(0.1, 15)

    # --- Derive ROAS (target variable) ---
    base_roas = 2.0
    roas = (
        base_roas
        + (retargeting == 1)             * 1.5
        + (creative_quality - 3)         * 0.4
        + (ad_formats == "video")        * 0.6
        + (objectives == "conversions")  * 0.8
        + (audience_sizes == "narrow")   * 0.5
        + (audience_sizes == "broad")    * -0.3
        + (platforms == "Meta")          * 0.3
        + (days_running / 90)            * 0.5
        + (ctr * 20)
        - (cpc / 5)
        + np.random.normal(0, 0.5, n_samples)
    ).clip(0.1, 12)

    df = pd.DataFrame({
        "platform":        platforms,
        "ad_format":       ad_formats,
        "objective":       objectives,
        "audience_size":   audience_sizes,
        "daily_budget":    daily_budget.round(2),
        "creative_quality": creative_quality,
        "days_running":    days_running,
        "retargeting":     retargeting,
        "day_of_week":     day_of_week,
        "ctr":             ctr.round(4),
        "cpc":             cpc.round(2),
        "roas":            roas.round(2),
    })

    return df


if __name__ == "__main__":
    df = generate_ad_data()
    df.to_csv("data/ad_campaigns.csv", index=False)
    print(f"Generated {len(df)} samples → data/ad_campaigns.csv")
    print(df.describe())
