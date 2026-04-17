# InsightForge AI — Customer Decision Engine

Production-style Streamlit dashboard for churn prediction, risk segmentation, intervention funnel analysis, and business-ready decision support.

## Overview
InsightForge AI turns churn model outputs into decision-focused analytics for product, growth, and retention teams.

The app:
- loads prebuilt XGBoost artifacts (`model/xgboost_model.pkl`, `model/features.pkl`)
- applies the same engineered features used during training
- predicts churn probability and churn risk by threshold
- segments users using RFM logic
- generates action recommendations per customer
- visualizes risk distribution, intervention funnel, and confusion metrics

## Project Structure
```text
InsightForge AI/
├── app.py
├── data/
│   ├── ecommerce_customer_churn_dataset.csv
│   ├── processed_data.csv
│   ├── sample_input.csv
│   └── model_outputs.csv
├── model/
│   ├── xgboost_model.pkl
│   └── features.pkl
├── notebooks/
│   └── analysis.ipynb
└── src/
```

## Core Features
- Churn probability inference with adjustable threshold
- Auto best-threshold support from labeled data
- Segment tagging from RFM score
- Executive decision summary with priority intervention buckets
- Customer Risk-to-Action funnel
- Confusion-matrix-driven evaluation block
- Downloadable prioritized action list

## Feature Engineering (Aligned With Training)
Engineered features used by model:
- `RFM_score`
- `engagement_score`
- `risk_score`
- `loyalty_score`
- `friction_score`
- `support_intensity`
- `value_per_purchase`

## Installation
### 1) Clone
```bash
git clone <YOUR_REPO_URL>
cd "InsightForge AI"
```

### 2) Create environment
```bash
python -m venv .venv
```

Activate:
- Windows (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
```
- macOS/Linux:
```bash
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

## Run App
```bash
streamlit run app.py
```

## Data Source Options in Dashboard
- Sample Input
- Preprocessed Data
- Raw Ecommerce Data

## Evaluation Metrics Shown
When labels are available (`Churned`/`churn`), app displays:
- confusion matrix (TP, FP, TN, FN)
- true positive rate (TPR)
- true negative rate (TNR)
- false positive rate (FPR)
- false negative rate (FNR)

## Suggested Demo Flow (Hackathon)
1. Start with `Raw Ecommerce Data`
2. Enable threshold optimization and explain selected threshold
3. Show churn distribution + risk banding
4. Present intervention funnel and priority buckets
5. Highlight confusion metrics and business tradeoff
6. Export prioritized actions

## Tech Stack
- Python
- Streamlit
- XGBoost
- scikit-learn
- pandas, numpy, matplotlib, joblib

## Notes
- Model artifacts are already included.
- If you retrain the model, overwrite files under `model/` and keep feature order consistent with `features.pkl`.

## Author
Built for the InsightForge AI Customer Analytics Decision Engine project.
