# RetainIQ — Fintech Retention Intelligence Platform

AI-powered fintech analytics platform for behavioral user segmentation, churn risk prediction, retention prioritization, and operational decision intelligence.

---

## Overview

RetainIQ transforms anonymized fintech behavioral event data into actionable retention intelligence.

The platform combines:
- behavioral segmentation
- churn probability prediction
- lifecycle intelligence
- intervention prioritization
- operational retention analytics

to help fintech platforms identify high-risk user cohorts and prioritize strategic retention actions.


---

## Problem Statement

### Blostem Challenge
Build an analytics engine that:
- segments fintech users by behavior
- predicts churn risk per segment
- identifies top churn drivers
- generates actionable retention intelligence

Expected deliverable:
- working analytics prototype
- behavioral segmentation
- churn prediction engine
- strategic retention insights

---

## Core Capabilities

### Behavioral Segmentation
Users are segmented using:
- RFM scoring
- lifecycle stage analysis
- product engagement patterns
- behavioral activity signals

### Churn Intelligence
The platform:
- predicts churn probability
- dynamically classifies churn risk
- prioritizes high-risk user cohorts
- tracks retention exposure

### Retention Decision Engine
RetainIQ converts churn analytics into:
- intervention priority buckets
- retention opportunity mapping
- strategic execution insights
- operational action guidance

### Executive Analytics Dashboard
Interactive Streamlit dashboard featuring:
- Executive Snapshot
- Segmentation Intelligence
- Churn Risk Engine
- Retention Action Funnel
- Retention Priority Quadrant
- Dynamic AI-generated insights

---

## Project Architecture

```text
RetainIQ/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── processed_fintech_data.csv
│   ├── fintech_scored_users.csv
│   ├── fintech_segment_scores.csv
│   └── fintech_top3_churn_hypotheses.csv
│
├── model/
│   ├── fintech_features.pkl
│   ├── fintech_xgboost_model.pkl
│   └── fintech_xgboost_calibrated_model.pkl
│
├── notebooks/
│   └── analysis.ipynb
│
└── src/
    ├── build_fintech_datasets.py
    ├── preprocessing.py
    ├── segmentation.py
    ├── prediction.py
    └── decision_engine.py
```

---

## Machine Learning Pipeline

### Feature Engineering
Key engineered behavioral features:
- RFM score
- engagement score
- support intensity
- activity trend
- transaction frequency
- transaction value
- lifecycle activity indicators

### Modeling
Models used:
- XGBoost Classifier
- Probability Calibration
- Threshold-based churn classification

### Segmentation Strategy
User cohorts are segmented using:
- value tier
- lifecycle stage
- product engagement mix
- behavioral retention signals

---

## Dashboard Modules

### Executive Snapshot
High-level portfolio churn overview with strategic KPIs and AI-generated business insights.

### Segmentation Intelligence
Behavioral cohort analysis across:
- RFM segments
- lifecycle stages
- product mix exposure
- churn concentration

### Churn Risk Engine
Interactive churn analytics including:
- probability distribution
- threshold calibration
- risk prioritization
- high-risk cohort tracking

### Retention Action Funnel
Operational funnel showing:
- total users
- high-risk users
- intervention candidates
- critical retention cohorts

### Retention Priority Quadrant
Business-oriented retention prioritization framework for strategic execution planning.

---

## Technology Stack

### Frontend & Dashboard
- Streamlit

### Machine Learning
- XGBoost
- scikit-learn

### Data Processing
- pandas
- numpy

### Visualization
- matplotlib

### AI Integration
- Google Gemini REST API
- Dynamic executive insight generation

---

## Installation

### 1. Clone Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd RetainIQ
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

Activate environment:

#### Windows
```powershell
.\.venv\Scripts\activate
```

#### macOS / Linux
```bash
source .venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key
```

---

## Run Application

```bash
streamlit run app.py
```

---

## Key Project Highlights

- Threshold-aware churn intelligence
- Behavioral user segmentation
- AI-generated executive insights
- Strategic retention prioritization
- Decision-oriented analytics design
- Operational fintech retention framework

---

## Design Philosophy

RetainIQ is designed around:
- operational intelligence
- executive readability
- behavioral analytics
- strategic retention decision-making
- focuses on actionable business intelligence rather than model-centric visualization.

The platform uses lightweight REST-based AI integration for scalable and deployment-safe insight generation.

---

## Future Enhancements

Potential future extensions:
- real-time event ingestion
- cohort drift monitoring
- intervention outcome tracking
- automated retention experimentation
- partner-level risk analytics

---

## Deployment

RetainIQ is deployed using Streamlit Cloud with environment-based secret management.

### Deployment Stack
- Streamlit Cloud
- GitHub Repository Integration
- Gemini REST API Integration

### Environment Variables

Configure the following secret inside Streamlit Cloud:

```toml
GEMINI_API_KEY = "your_api_key"
```

### Launch Application

```bash
streamlit run app.py
```

---

## Author

Garv Chanana
