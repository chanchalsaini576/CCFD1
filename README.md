# Explainable AI-Driven Credit Card Fraud Detection System with Real-Time Risk Monitoring

**Govt. Engineering College, Ajmer — Dept. of CSE — B.Tech VII Semester, Project-I**

A machine learning system that detects fraudulent credit card transactions,
with a live web dashboard showing predictions, confidence scores, a
simulated real-time transaction feed, and feature-level explanations (SHAP)
for flagged transactions.

## Problem

Fraud detection is a **severe class-imbalance problem** — in real-world
data, fraud typically makes up less than 0.5% of all transactions. A model
that always predicts "not fraud" would score ~99.5% accuracy while catching
zero fraud cases. This project is built around handling that imbalance
correctly and evaluating with the right metrics (Precision, Recall, F1,
AUC-ROC) instead of accuracy.

## Project Structure

```
credit-card-fraud-detection/
├── data/
│   ├── raw/creditcard.csv          # raw dataset (Time, V1-V28, Amount, Class)
│   └── processed/                   # train/test splits after SMOTE + scaling
├── src/
│   ├── preprocessing.py             # scaling, train/test split, SMOTE
│   ├── train_model.py                # trains & compares 3 models
│   └── explainability.py             # SHAP-based feature importance
├── app/
│   ├── app.py                        # Flask backend + API routes
│   ├── templates/dashboard.html      # dashboard UI
│   └── static/shap_summary.png       # global feature-importance chart
├── models/
│   ├── fraud_model.pkl               # trained model (XGBoost)
│   ├── feature_names.pkl
│   └── results_summary.json          # metrics for all 3 models compared
├── requirements.txt
└── README.md
```

## Dataset

This project uses the same schema as the well-known Kaggle
**"Credit Card Fraud Detection" (ULB)** dataset: `Time`, 28 PCA-transformed
features (`V1`–`V28`), `Amount`, and `Class` (0 = legitimate, 1 = fraud),
with a realistic ~0.5% fraud rate.

> **Note:** Because this environment has no internet access to Kaggle, a
> synthetic dataset matching the exact same schema and imbalance ratio was
> generated in `generate_synthetic_data.py` for local development. **To use
> the real dataset**, download `creditcard.csv` from
> [Kaggle: mlg-ulb/creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)
> and drop it into `data/raw/creditcard.csv` — no code changes needed, since
> the column schema is identical.

## How It Works

### 1. Preprocessing (`src/preprocessing.py`)
- Scales `Amount` and `Time` (the only non-PCA-transformed columns).
- Splits into train/test **before** any resampling.
- Applies **SMOTE** (Synthetic Minority Oversampling) only to the training
  set. Applying SMOTE before the split, or to the test set, leaks synthetic
  patterns into evaluation and inflates metrics artificially — this is a
  common mistake worth explaining in your viva.

### 2. Model Training (`src/train_model.py`)
Trains and compares three models:
- Logistic Regression (baseline)
- Random Forest
- XGBoost

Evaluated on **Precision, Recall, F1-score, PR-AUC (average precision), and
ROC-AUC** on the untouched test set. PR-AUC matters more than ROC-AUC here
because with <1% fraud, ROC-AUC can look deceptively high even for a
mediocre model — PR-AUC is more sensitive to how well the model does on the
rare, minority (fraud) class specifically. The best model is selected by
**F1-score** (balances catching fraud vs. not over-flagging legitimate
transactions) and saved to `models/fraud_model.pkl`.

### 3. Explainability (`src/explainability.py`)
Uses **SHAP (SHapley Additive exPlanations)** to show:
- Global feature importance (which features matter most overall) → saved
  as a bar chart.
- Per-transaction explanations (why *this specific* transaction was
  flagged) → shown live in the dashboard.

### 4. Web Dashboard (`app/app.py` + `app/templates/dashboard.html`)
A Flask app with three API endpoints:
- `/api/stats` — summary numbers (total scanned, fraud caught, recall %).
- `/api/transactions` — a sample of transactions with prediction,
  confidence score, and top contributing features for flagged ones.
- `/api/live-feed` — **real-time risk monitoring**: returns one freshly
  scored transaction (drawn from the held-out test set, so the model has
  never seen it during training) each time it's called. The dashboard
  polls this every 4 seconds and streams new rows into a live feed panel,
  simulating transactions arriving in real time at a bank.

The dashboard displays this as a live-styled table with color-coded
verdicts, confidence bars, and a scrolling live feed panel.

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Place the real Kaggle dataset at data/raw/creditcard.csv
#    A synthetic one is already included for immediate testing.

# 3. Run preprocessing
python3 src/preprocessing.py

# 4. Train models
python3 src/train_model.py

# 5. Generate explainability artifacts
python3 src/explainability.py

# 6. Launch the dashboard
cd app
python3 app.py
# Visit http://127.0.0.1:5000
```

## Deploying a Live Demo (100% Free, No Credit Card)

Render.com's free tier lets you host this Flask app with a real public URL,
free forever, no card required. (The only tradeoff: a free instance goes
to sleep after ~15 minutes of no visits, and takes 30–60 seconds to "wake
up" on the next request — totally fine for a viva demo, just click the
link a minute before you present.)

**Steps:**

1. **Push this project to GitHub.**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```
   Create a new empty repo on github.com, then:
   ```bash
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git branch -M main
   git push -u origin main
   ```

2. **Sign up at [render.com](https://render.com)** using your GitHub account
   (no card needed).

3. Click **New +** → **Web Service**, and connect your GitHub repo. Render
   will detect `render.yaml` in this project automatically and pre-fill:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn --chdir app app:app --bind 0.0.0.0:$PORT`
   - **Plan:** Free

   If it doesn't auto-detect, just paste those two commands manually and
   pick the **Free** instance type.

4. Click **Create Web Service**. First deploy takes 3–5 minutes (installing
   `xgboost`, `shap`, etc.). You'll get a live URL like
   `https://fraud-watch-dashboard.onrender.com`.

5. That's it — share that link. The model/data files (`models/`,
   `data/processed/`) are already committed in the repo, so the deployed
   app serves predictions immediately with no extra setup on Render's side.

**Alternative free options** if you want to compare: PythonAnywhere (free
tier, no card, but manual file upload instead of Git deploy) and Hugging
Face Spaces (free, works well if you're open to wrapping the app in a
Docker Space instead of a plain web service). Render is the simplest path
for a plain Flask app like this one.

## Results (on included synthetic dataset)



See `models/results_summary.json` for full metrics. On the real Kaggle
dataset, expect more realistic numbers — typically 85–95% recall and
high but non-perfect precision, since real fraud patterns overlap more
with legitimate transactions than this synthetic data does.

## Key Talking Points for Viva / Report

1. **Why not accuracy?** With <1% fraud, accuracy is meaningless — a
   do-nothing model scores ~99%. Precision/Recall/F1/AUC-ROC on the
   minority (fraud) class are what matter.
2. **Why SMOTE only on training data?** To avoid data leakage — the test
   set must reflect the real, imbalanced world the model will face in
   production.
3. **Precision vs. Recall tradeoff:** A missed fraud case (false negative)
   is usually more costly than a false alarm (false positive), so recall
   is often prioritized — but too many false positives erodes trust and
   creates review overhead. F1 balances both.
4. **Why explainability matters:** In finance, a "black box" flag isn't
   enough — analysts need to know *why* a transaction was flagged, both
   for trust and for regulatory/audit reasons.

   

## Next Steps (8th Sem Major Project Extension)

- Real-time stream processing (Kafka/Redis Streams) instead of batch
  predictions.
- Unsupervised anomaly detection (Isolation Forest/Autoencoders) to catch
  novel fraud patterns not seen in training.
- Analyst feedback loop + periodic model retraining.
- Docker containerization + cloud deployment for a live demo link.
