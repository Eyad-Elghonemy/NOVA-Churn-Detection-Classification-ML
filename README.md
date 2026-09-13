<div align="center">

<img src="nova_logo.png" alt="NOVA Logo" width="160" />

# NOVA — Churn Detection & Classification
**A production-grade ML service for predicting bank customer churn, served through a FastAPI backend and a Streamlit analytics dashboard.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-189fdd?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Container-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Pydantic](https://img.shields.io/badge/Pydantic-Validation-E92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![HuggingFace](https://img.shields.io/badge/Deployed-HuggingFace-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)

</div>

---

## 🔗 Live Demo

| Component | URL |
|---|---|
| **API (FastAPI — Hugging Face Spaces)** | https://eyadzz-churn-live.hf.space/ |
| **Dashboard (Streamlit — Churn Ledger)** | https://churnlive-6mnfachkcjvec7gfaqhfjb.streamlit.app/ |
| **Interactive API Docs (Swagger)** | https://eyadzz-churn-live.hf.space/docs |

> ⚠️ **Security note:** the key below is a **demo key** shared for evaluation only. If this repository is public, rotate it immediately (set a new `SECRET_KEY_TOKEN` in your Hugging Face Space secrets) — never rely on a key that has appeared in a public README for anything beyond a quick demo.

```
Demo X-API-Key: c0c2d9d05029aed5d5174ff5ff8e6d88
```

---

## What is NOVA?

NOVA is a full-stack machine learning service that predicts whether a bank customer is likely to **churn** (leave the bank), based on the classic `Churn_Modelling.csv` dataset (~10,000 customers, ~20% churn rate). It covers the entire pipeline end to end:

| Stage | What happens |
|---|---|
| 📊 **EDA & Preprocessing** | Cleaning, feature selection, encoding, scaling |
| ⚖️ **Imbalance handling** | Class weighting + SMOTE oversampling |
| 🧠 **Modeling** | Logistic Regression baseline → tuned Random Forest & XGBoost |
| 🚀 **Serving** | FastAPI service exposing both tuned models behind an API key |
| 📺 **Interface** | Streamlit dashboard ("Churn Ledger") for scoring, batch analysis, and model insights |

---

## Architecture

```
┌─────────────────────┐        HTTPS + X-API-Key          ┌───────────────────────┐
│   Streamlit Cloud   │ ─────────────────────────────▶   │  Hugging Face Spaces  │
│  "Churn Ledger"     │ ◀─────────────────────────────   │  FastAPI (main.py)    │
│  streamlit_app.py   │        JSON prediction            │  + Docker             │
└─────────────────────┘                                   └──────────┬────────────┘
                                                                     │
                                                         ┌───────────▼────────────┐
                                                         │  models/*.pkl          │
                                                         │  preprocessor          │
                                                         │  forest_tuned          │
                                                         │  xgb-tuned             │
                                                         └────────────────────────┘
```

---

## Features

| Feature | Detail |
|---|---|
| 🧠 **Dual model serving** | Random Forest and XGBoost available as separate endpoints |
| 🔐 **API key authentication** | All prediction endpoints require `X-API-Key` header |
| 📊 **Portfolio analytics** | Dataset-level KPIs, geography/gender breakdowns, churn rate |
| 🎯 **Single-customer scoring** | Live risk gauge + top model feature drivers |
| 🗂️ **Batch scoring** | Upload a CSV, score all customers, download results |
| 📈 **Model insights** | Development journey, leaderboard of all configs tried, feature importance |
| 🐳 **Dockerized** | Same container runs locally and on Hugging Face Spaces |

---

## Project Structure

```
churn_live/
├── main.py                    # FastAPI app & routes
├── requirements.txt           # Backend dependencies
├── Dockerfile                 # Container spec for Hugging Face Spaces
├── streamlit_app.py           # Streamlit dashboard ("Churn Ledger")
├── requirements_streamlit.txt # Dashboard dependencies
├── .streamlit/
│   └── config.toml            # Dashboard theme
├── utils/
│   ├── __init__.py
│   ├── config.py              # Env vars + model loading
│   ├── CustomerData.py        # Pydantic request schema
│   └── inference.py           # Prediction logic
├── models/
│   ├── preprocessor.pkl
│   ├── forest_tuned.pkl
│   └── xgb-tuned.pkl
├── notebooks/
│   └── notebook.ipynb         # Full training & EDA workflow
└── README.md
```

---

## Model Performance

| # | Model | Configuration | Test F1 |
|---|---|---|---|
| 1 | Logistic Regression | Baseline | 0.375 |
| 2 | Logistic Regression | Class-weighted / SMOTE | ~0.50 |
| 3 | Random Forest | Class-weighted / SMOTE | 0.57 – 0.59 |
| 4 | **Random Forest** | **Tuned (GridSearchCV)** | **0.623 ⭐** |
| 5 | XGBoost | Base | 0.595 |
| 6 | XGBoost | Tuned (RandomizedSearchCV) | 0.609 |

Both tuned models (#4 and #6) are deployed in production. Random Forest currently holds the best test-set F1; XGBoost is offered as an alternative for side-by-side comparison.

**Top predictive features (Random Forest):** `Age` › `NumOfProducts` › `Balance` › `IsActiveMember` › `Geography`

---

## API Reference

**Base URL:** `https://eyadzz-churn-live.hf.space`

All prediction endpoints require an `X-API-Key` header.

---

### `GET /`
Health check — no authentication required.

**Response**
```json
{ "Message": "Welcome To My Churn-Detection API v1.0" }
```

---

### `POST /predict/forest`
### `POST /predict/xgboost`

Predict churn probability using the Random Forest or XGBoost model.

**Headers**

| Header | Required | Value |
|---|---|---|
| `X-API-Key` | ✅ | Your secret key |
| `Content-Type` | ✅ | `application/json` |

**Request body**

```json
{
  "CreditScore": 650,
  "Geography": "France",
  "Gender": "Male",
  "Age": 38,
  "Tenure": 5,
  "Balance": 75000.0,
  "NumOfProducts": 1,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 100000.0
}
```

**Field constraints**

| Field | Type | Constraints |
|---|---|---|
| `CreditScore` | int | — |
| `Geography` | string | `France`, `Spain`, `Germany` |
| `Gender` | string | `Male`, `Female` |
| `Age` | int | 18 – 100 |
| `Tenure` | int | 0 – 10 |
| `Balance` | float | ≥ 0 |
| `NumOfProducts` | int | 1 – 4 |
| `HasCrCard` | int | 0 or 1 |
| `IsActiveMember` | int | 0 or 1 |
| `EstimatedSalary` | float | ≥ 0 |

**Success response**
```json
{
  "Churn_Prediction": false,
  "Churn_Probability": 0.258
}
```

**Example (cURL)**
```bash
curl -X POST "https://eyadzz-churn-live.hf.space/predict/forest" \
  -H "X-API-Key: c0c2d9d05029aed5d5174ff5ff8e6d88" \
  -H "Content-Type: application/json" \
  -d '{
        "CreditScore": 650, "Geography": "France", "Gender": "Male",
        "Age": 38, "Tenure": 5, "Balance": 75000.0,
        "NumOfProducts": 1, "HasCrCard": 1, "IsActiveMember": 1,
        "EstimatedSalary": 100000.0
      }'
```

---

## Environment Variables

The API reads configuration from a `.env` file via `utils/config.py`.

```bash
cp .env.example .env
```

**`.env.example`**
```env
APP_NAME="Churn-Detection"
VERSION="1.0"
SECRET_KEY_TOKEN="your-secret-key-here"
```

| Variable | Purpose |
|---|---|
| `APP_NAME` | App title in FastAPI docs and the `/` welcome message |
| `VERSION` | API version shown in docs and the `/` welcome message |
| `SECRET_KEY_TOKEN` | Value every request must send in the `X-API-Key` header |

`.env` is listed in `.gitignore` and is never committed. When deploying to **Hugging Face Spaces**, set these three variables under **Settings → Variables and secrets** instead of uploading a `.env` file.

---

## Installation & Running Locally

### 1. Clone and install

```bash
git clone <your-repo-url>
cd churn_live
pip install -r requirements.txt
cp .env.example .env   # fill in APP_NAME, VERSION, SECRET_KEY_TOKEN
```

### 2. Backend (FastAPI)

```bash
uvicorn main:app --reload
```

Available at: `http://localhost:8000`
Swagger UI: `http://localhost:8000/docs`

### 3. Dashboard (Streamlit — Churn Ledger)

```bash
pip install -r requirements_streamlit.txt
streamlit run streamlit_app.py
```

Open the sidebar and set:
- **API base URL** → `http://localhost:8000` (or the deployed URL)
- **X-API-Key** → the same value as `SECRET_KEY_TOKEN`

---

## Docker

The `Dockerfile` builds the exact environment used in production on Hugging Face Spaces.

```bash
docker build -t nova-churn-api.
docker run --rm --env-file .env -p 8000:7860 nova-churn-api
```

- The container listens on **port 7860** (required by Hugging Face Spaces) — the command above maps it to `8000` locally.
- `--env-file .env` passes `APP_NAME`, `VERSION`, and `SECRET_KEY_TOKEN` into the container.
- Once running, the API is identical to the hosted version: `http://localhost:8000/docs`

On Hugging Face Spaces, the same `Dockerfile` is picked up automatically on every push — no extra configuration needed beyond the repository secrets.

---

## Deployment

| Component | Platform | Notes |
|---|---|---|
| API | Hugging Face Spaces (Docker SDK) | Free tier — ~30–60s cold start after inactivity |
| Dashboard | Streamlit Community Cloud | Free tier |

Environment variables are set as **repository secrets** on the Space — never committed to the repo.

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **ML** | scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Dashboard** | Streamlit, Plotly, Pandas |
| **Deployment** | Docker, Hugging Face Spaces, Streamlit Community Cloud |

---

<div align="center">
<sub>NOVA · built with FastAPI, scikit-learn, XGBoost, Streamlit &amp; Docker</sub>
</div>
