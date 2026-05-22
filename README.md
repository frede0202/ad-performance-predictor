# Ad Performance Predictor

A machine learning project that predicts ROAS (Return on Ad Spend) for digital ad campaigns.

## The Problem

E-commerce brands running ads on Meta and Google often don't know which campaign setup will perform best until they've already spent their budget. This project explores whether we can **predict ROAS before spending** based on campaign configuration alone.

## Approach

- Generate realistic synthetic ad campaign data (2,000 samples) with features mirroring real Meta/Google Ads setups
- Train an XGBoost regression model to predict ROAS
- Identify which factors drive performance most (feature importance)
- Score new campaign variants before launch to rank them by predicted ROAS

## Project Structure

```
ad_performance_predictor/
├── generate_data.py   # Generates synthetic campaign dataset
├── train.py           # Trains and evaluates the XGBoost model
├── predict.py         # Scores new campaign variants
├── requirements.txt
├── data/
│   └── ad_campaigns.csv
└── outputs/
    ├── feature_importance.png
    ├── predicted_vs_actual.png
    └── metrics.json
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
mkdir data
python generate_data.py

# 3. Train model and evaluate
python train.py

# 4. Predict ROAS for new campaign variants
python predict.py
```

## Results

The model achieves strong predictive performance on held-out test data:

| Metric | Value |
|--------|-------|
| MAE    | ~0.45 |
| R²     | ~0.82 |

**Key drivers of ROAS (from feature importance):**

1. `retargeting` — retargeting audiences consistently outperform cold audiences
2. `creative_quality` — higher quality creatives drive significantly better returns
3. `ctr` — click-through rate is a strong signal of audience-creative fit
4. `ad_format` — video outperforms image and carousel on average
5. `objective` — conversion-optimised campaigns outperform traffic/awareness

## How This Applies to Marketer.com

Marketer's Manta product analyses campaign data to surface actionable insights. A model like this could power:

- **Pre-launch ranking** of creative variants before spending budget
- **Budget allocation recommendations** across platforms and formats
- **Automated alerts** when a campaign's predicted ROAS drops below threshold
- **A/B test prioritisation** by scoring variants before running them

With access to real campaign data the model would be retrained continuously, improving as more signals come in — turning it from a predictor into a live optimisation engine.

## Notes

This project uses synthetic data to demonstrate the ML approach. In a production setting, the same pipeline would connect to live Meta/Google Ads APIs and retrain on real performance data.
