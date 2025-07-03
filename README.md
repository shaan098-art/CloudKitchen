
# Cloud Kitchen Consumer Insights Dashboard

A full‑featured Streamlit application that explores the synthetic cloud‑kitchen survey, delivers ML‑driven insights, and allows interactive experimentation.

## Features
* **Data Visualisation** – 10+ descriptive plots with filters.
* **Classification** – KNN, Decision Tree, Random Forest, Gradient Boosting with metrics table, confusion matrix toggle, ROC comparison, and batch inference upload/download.
* **Clustering** – K‑Means with elbow chart, adjustable cluster slider, persona table, and data export.
* **Association Rules** – Apriori mining with configurable support/confidence and top‑10 rule display.
* **Regression** – Ridge, Lasso, Decision Tree regressors with metrics and actual‑vs‑predicted visual.

## Running locally
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud deployment
1. Push this repository (including `data/cloud_kitchen_survey_descriptive.csv` or set `csv_url` secret) to GitHub.
2. In Streamlit Cloud, create a new app pointing to `app.py`.
3. (Optional) Add a `csv_url` secret to the *Secrets* manager if the dataset lives elsewhere (e.g. a GitHub raw link).

## Repository layout
```
.
├── app.py                # Main entrypoint
├── pages/                # Multi‑page modules
├── utils.py              # Shared helpers
├── requirements.txt
└── README.md
```

Enjoy exploring your data‑ready cloud‑kitchen universe! 🍲
