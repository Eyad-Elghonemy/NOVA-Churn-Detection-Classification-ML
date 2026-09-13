"""
Churn Ledger — a Streamlit control room for the Churn-Detection FastAPI service.

Run alongside the FastAPI backend:
    uvicorn main:app --reload          # backend, from the API project root
    streamlit run streamlit_app.py     # this dashboard

The app never loads models directly — every prediction is a real HTTP call to
your FastAPI instance, so it reflects exactly what the API returns.
"""

import hashlib
import json
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import base64
import io
from PIL import Image

# Embedded NOVA logo (base64 PNG, cropped & resized ~235x240) — no external asset file needed
NOVA_LOGO_B64 = "iVBORw0KGgoAAAANSUhEUgAAAOsAAADwCAYAAADhG5ONAABeXElEQVR42u2deXxcVd3/P99zzr0zkz1tmu4blK2FsrSsiklYZBVBTABBUUBQEBV31J/JgBv48Pjw+AiCuACCmgFkKbuShH1p2Ru6703aZl9mufeec76/P+5MGipg2Snc96vzmnQySSaT87nf9XwPEBEREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREREfGRhhn0Ufy9RfSnj9jRuPbac1Uk1oiIHcCqBh1LY5FYIyI+4KQa6kUxEFnWiIgPPLMhJVw3EmtExAeeLa4NjJP/D0VijYj4oMWqeWHm4BbFnFCsHFnWiIgPploBIK5yxds+9lFBRasg4t2gublephf7pWvQP5xMtum3+/2amkAAWEmaIJi7t/frGhshPjVpnpx/3qIgsqwREa9BQ0PKAP25GSiu+lNjTcW27uxbXrCCxzgJ3i7xtzTWqHmTjo/PO3eR/jC8p5FljXjX+FKyLQdg0+8bPznmj5d+cippr4vCx+jNOrHJZPj8hOBKJgrFR28s1KHS0opcZaKP6MPhMEeWNeJd58vJB3oD3dMjY7Epf730k1MLQn2TbYMMgIREWczm3tClXXjNPCddIiaXDg31hxY+ilkjIrYXOi+5KANgxZ8aa2Y0/+yw2TlXriV6MM0MIhoR4+t+PQC+/1dHFsXd4erioph5DcNKALi5uV72btw8qzhh19d9t02/FSseWdaIjzI8yjVeEwi/0/X8vZoba2blXVRubPz3tViIbzn/1eXlA5Uxx461OsvbfF8CwMygys7NcxMxu6nua23DHyahRmKNeO9Vy6DTL36075Wg7WmKc+mtv/zEkQt+8fHKZBKWGTTaNaZQgCKVqhcAEANXFcd4TElJCbi5Xo5ypZmZqeU3Hz8kIfTmQ89/tK/w+IfpvYvEGvHe+sOhJaVkErb+4oefc2CWapJHLfjVxw8kAhOFrmzh+alUPe3Ut0oAQMUYMam8ksbkTJ9Y1LdKcCNE4fs9c83Bx8eF3vzxCx/vyLvWH7oqLEXLJ+L9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"

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
NOVA_FAVICON = Image.open(io.BytesIO(base64.b64decode(NOVA_LOGO_B64)))

st.set_page_config(
    page_title="NOVA",
    page_icon=NOVA_FAVICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Backend connection (built-in, not user-editable)
# --------------------------------------------------------------------------
API_BASE_URL = "https://eyadzz-churn-live.hf.space/"

# --------------------------------------------------------------------------
# Design tokens & theme
# --------------------------------------------------------------------------
INK = "#10161C"
PANEL = "#171F27"
PANEL_ALT = "#1D2731"
BORDER = "rgba(201,162,39,0.18)"
BRASS = "#C9A227"
BRASS_DIM = "#8A7220"
TEAL = "#3FA796"
ALERT = "#C1443C"
TEXT = "#EDEDE7"
MUTED = "#8B93A0"

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT, size=13),
        colorway=[TEAL, BRASS, ALERT, "#6E8CA0", "#5B4B8A"],
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
)

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background: radial-gradient(circle at 15% 0%, #141B23 0%, {INK} 45%) fixed;
    color: {TEXT};
}}

section[data-testid="stSidebar"] {{
    background: {PANEL};
    border-right: 1px solid {BORDER};
}}

h1, h2, h3 {{
    font-family: 'Fraunces', serif;
    letter-spacing: -0.01em;
}}

.ledger-title {{
    font-family: 'Fraunces', serif;
    font-size: 2.1rem;
    font-weight: 600;
    color: {TEXT};
    margin-bottom: 0;
}}
.ledger-subtitle {{
    color: {MUTED};
    font-size: 0.95rem;
    margin-top: 0.15rem;
    letter-spacing: 0.02em;
}}
.eyebrow {{
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.72rem;
    color: {BRASS};
    font-weight: 600;
}}
.rule {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 0.6rem 0 1.4rem 0;
}}

.kpi-card {{
    background: linear-gradient(180deg, {PANEL_ALT} 0%, {PANEL} 100%);
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 1rem 1.2rem;
    height: 100%;
}}
.kpi-label {{
    color: {MUTED};
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
.kpi-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem;
    font-weight: 600;
    color: {TEXT};
    margin-top: 0.15rem;
}}
.kpi-delta {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: {TEAL};
}}

.badge {{
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}}
.badge-low {{ background: rgba(63,167,150,0.15); color: {TEAL}; border: 1px solid rgba(63,167,150,0.4); }}
.badge-med {{ background: rgba(201,162,39,0.15); color: {BRASS}; border: 1px solid rgba(201,162,39,0.4); }}
.badge-high {{ background: rgba(193,68,60,0.15); color: {ALERT}; border: 1px solid rgba(193,68,60,0.4); }}

.ledger-row {{
    border-bottom: 1px solid {BORDER};
    padding: 0.55rem 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
}}

.stButton>button {{
    background: {BRASS};
    color: {INK};
    border: none;
    font-weight: 600;
    border-radius: 4px;
}}
.stButton>button:hover {{
    background: {BRASS_DIM};
    color: {TEXT};
}}

div[data-testid="stMetricValue"] {{
    font-family: 'IBM Plex Mono', monospace;
}}

/* --- Sidebar navigation menu --- */
section[data-testid="stSidebar"] div[data-testid="stButton"] button {{
    background: transparent;
    color: {MUTED};
    border: 1px solid transparent;
    border-left: 3px solid transparent;
    text-align: left;
    justify-content: flex-start;
    font-weight: 500;
    font-size: 0.95rem;
    padding: 0.6rem 0.8rem;
    border-radius: 6px;
    margin-bottom: 0.2rem;
    width: 100%;
    box-shadow: none;
}}
section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {{
    background: rgba(201,162,39,0.08);
    color: {TEXT};
    border-left-color: rgba(201,162,39,0.35);
}}
section[data-testid="stSidebar"] div[data-testid="stButton"] button:focus:not(:active) {{
    color: {TEXT};
    border-left-color: rgba(201,162,39,0.35);
}}
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {{
    background: linear-gradient(90deg, rgba(201,162,39,0.20), rgba(201,162,39,0.04));
    color: {BRASS} !important;
    border-left: 3px solid {BRASS};
    font-weight: 700;
}}
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"]:hover {{
    background: linear-gradient(90deg, rgba(201,162,39,0.26), rgba(201,162,39,0.08));
    color: {BRASS} !important;
}}
.nav-eyebrow-spacer {{ margin-bottom: 0.4rem; }}

/* --- KPI cards v2 (Portfolio Overview) --- */
.kpi-card-v2 {{
    background: linear-gradient(155deg, {PANEL_ALT} 0%, {PANEL} 100%);
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    position: relative;
    overflow: hidden;
    transition: transform .18s ease, border-color .18s ease;
    height: 100%;
}}
.kpi-card-v2:hover {{
    transform: translateY(-3px);
    border-color: rgba(201,162,39,0.45);
}}
.kpi-card-v2 .kpi-accent-bar {{
    position: absolute; top: 0; left: 0; bottom: 0; width: 4px;
}}
.kpi-card-v2 .kpi-icon {{ font-size: 1.35rem; opacity: 0.9; }}
.kpi-card-v2 .kpi-label-v2 {{
    color: {MUTED};
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-top: 0.55rem;
}}
.kpi-card-v2 .kpi-value-v2 {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: {TEXT};
    margin-top: 0.1rem;
}}

/* --- Leaderboard rows (Model Insights) --- */
.rank-row {{
    display: flex;
    align-items: center;
    gap: 0.9rem;
    padding: 0.5rem 0.1rem;
    border-bottom: 1px solid {BORDER};
}}
.rank-num {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem;
    color: {MUTED};
    width: 1.5rem;
}}
.rank-num.gold {{ color: {BRASS}; font-weight: 700; }}
.rank-bar-track {{
    flex: 1;
    background: rgba(255,255,255,0.05);
    border-radius: 4px;
    height: 10px;
    overflow: hidden;
}}
.rank-bar-fill {{ height: 100%; border-radius: 4px; }}
.rank-label {{ min-width: 230px; font-size: 0.84rem; color: {TEXT}; }}
.rank-score {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.84rem;
    color: {MUTED};
    width: 3.4rem;
    text-align: right;
}}

/* --- NOVA logo: real artwork, with a soft ambient glow behind it --- */
.nova-logo-row {{
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 0.1rem;
}}
.nova-logo-wrap {{
    position: relative;
    flex: none;
    display: flex;
    align-items: center;
    justify-content: center;
}}
.nova-logo-wrap::before {{
    content: "";
    position: absolute;
    inset: -30%;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(201,162,39,0.20) 0%, rgba(63,167,150,0.12) 45%, transparent 72%);
    filter: blur(4px);
    animation: nova-glow-pulse 3.5s ease-in-out infinite;
    z-index: 0;
}}
@keyframes nova-glow-pulse {{
    0%, 100% {{ opacity: 0.55; transform: scale(0.92); }}
    50%      {{ opacity: 1; transform: scale(1.08); }}
}}
.nova-logo-img {{
    position: relative;
    z-index: 1;
    width: 100%;
    height: 100%;
    object-fit: contain;
    filter: drop-shadow(0 0 6px rgba(201,162,39,0.25));
}}
.nova-logo-link {{
    text-decoration: none !important;
    color: inherit !important;
    cursor: pointer;
    display: inline-block;
    border-radius: 10px;
    transition: transform 0.15s ease, filter 0.15s ease;
}}
.nova-logo-link:hover {{
    transform: translateY(-1px);
    filter: brightness(1.12);
}}
.nova-logo-link:hover .nova-logo-img {{
    filter: drop-shadow(0 0 10px rgba(201,162,39,0.45));
}}
.nova-wordmark {{
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 1.7rem;
    letter-spacing: 0.06em;
    background-image: linear-gradient(100deg, {TEAL} 0%, {BRASS} 65%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}}
.nova-wordmark-sub {{
    color: {MUTED};
    font-size: 0.72rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-top: -0.2rem;
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Reference dataset facts (from the training notebook — Churn_Modelling.csv)
# Used for the portfolio-level dashboard since the raw dataset isn't bundled
# with the API. Swap in a live query if you wire the API to a database.
# --------------------------------------------------------------------------
DATASET_STATS = {
    "total_customers": 10000,
    "churn_rate": 0.2037,
    "geography": {"France": 5014, "Germany": 2509, "Spain": 2477},
    "gender": {"Male": 5457, "Female": 4543},
    "avg_salary_by_gender": {"Female": 100576, "Male": 99672},
    "train_size": 7990,
    "test_size": 1998,
}

FEATURE_IMPORTANCE = {
    "Age": 0.37468,
    "NumOfProducts": 0.24874,
    "Balance": 0.09514,
    "IsActiveMember": 0.06470,
    "Geography_Germany": 0.05457,
    "EstimatedSalary": 0.05183,
    "CreditScore": 0.04889,
    "Tenure": 0.02671,
    "Gender_Male": 0.02164,
    "Geography_Spain": 0.00683,
    "HasCrCard": 0.00629,
}

MODEL_LEDGER = [
    {"entry": "Logistic Regression", "note": "baseline, no balancing", "train_f1": 0.309, "test_f1": 0.375},
    {"entry": "Logistic Regression", "note": "class-weighted", "train_f1": 0.498, "test_f1": 0.499},
    {"entry": "Logistic Regression", "note": "SMOTE oversampled", "train_f1": 0.498, "test_f1": 0.508},
    {"entry": "Random Forest", "note": "class-weighted", "train_f1": 0.600, "test_f1": 0.573},
    {"entry": "Random Forest", "note": "SMOTE oversampled", "train_f1": 0.618, "test_f1": 0.590},
    {"entry": "Random Forest", "note": "tuned (GridSearchCV)", "train_f1": 0.680, "test_f1": 0.623},
    {"entry": "XGBoost", "note": "base model", "train_f1": 0.703, "test_f1": 0.595},
    {"entry": "XGBoost", "note": "tuned (RandomizedSearchCV)", "train_f1": 0.624, "test_f1": 0.609},
]

CUSTOMER_FIELDS_HELP = {
    "CreditScore": "Bureau credit score, typically 300–850.",
    "Geography": "Customer's country of residence.",
    "Gender": "Customer's gender as recorded.",
    "Age": "Customer age in years (18–100).",
    "Tenure": "Years as a customer of the bank (0–10).",
    "Balance": "Current account balance.",
    "NumOfProducts": "Number of bank products held (1–4).",
    "HasCrCard": "Whether the customer holds a credit card.",
    "IsActiveMember": "Whether the customer is an active member.",
    "EstimatedSalary": "Estimated annual salary.",
}

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: timestamp, model, prob, pred, inputs

NAV_ITEMS = [
    ("📊", "Portfolio Overview"),
    ("🧾", "Score a Customer"),
    ("📥", "Batch Scoring"),
    ("📚", "Model Insights"),
    ("📈", "Usage Analytics"),
]
if "nav_page" not in st.session_state:
    st.session_state.nav_page = NAV_ITEMS[0][1]

# Clicking the NOVA logo/wordmark sends ?nav=home — catch it here and jump
# to Portfolio Overview, then clear the query param so it doesn't stick around.
if st.query_params.get("nav") == "home":
    st.session_state.nav_page = "Portfolio Overview"
    st.query_params.clear()

# --------------------------------------------------------------------------
# API helpers
# --------------------------------------------------------------------------
def api_health(base_url: str, timeout=3):
    try:
        r = requests.get(base_url.rstrip("/") + "/", timeout=timeout)
        if r.status_code == 200:
            return True, r.json().get("Message", "Connected")
        return False, f"HTTP {r.status_code}"
    except requests.exceptions.RequestException as e:
        return False, str(e)


def call_predict(base_url: str, api_key: str, model: str, payload: dict, timeout=8):
    endpoint = "/predict/forest" if model == "Random Forest" else "/predict/xgboost"
    url = base_url.rstrip("/") + endpoint
    headers = {"X-API-Key": api_key}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if r.status_code == 200:
            return True, r.json()
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        return False, f"HTTP {r.status_code}: {detail}"
    except requests.exceptions.RequestException as e:
        return False, str(e)


def get_usage_stats(base_url: str, api_key: str, timeout=8):
    url = base_url.rstrip("/") + "/stats"
    headers = {"X-API-Key": api_key}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code == 200:
            return True, r.json()
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        return False, f"HTTP {r.status_code}: {detail}"
    except requests.exceptions.RequestException as e:
        return False, str(e)


def risk_band(prob: float):
    if prob < 0.33:
        return "low", "badge-low", TEAL
    if prob < 0.66:
        return "medium", "badge-med", BRASS
    return "high", "badge-high", ALERT


def gauge_chart(prob: float):
    band, _, color = risk_band(prob)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 44, "family": "IBM Plex Mono"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 1,
                "bordercolor": BORDER,
                "steps": [
                    {"range": [0, 33], "color": "rgba(63,167,150,0.18)"},
                    {"range": [33, 66], "color": "rgba(201,162,39,0.18)"},
                    {"range": [66, 100], "color": "rgba(193,68,60,0.18)"},
                ],
                "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.9, "value": prob * 100},
            },
        )
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=280, margin=dict(l=20, r=20, t=30, b=10))
    return fig


def kpi_card(label, value, col):
    col.markdown(
        f"""<div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def nova_logo(sub_label="Churn Detection Service", size=46, clickable=True):
    inner = f"""<div class="nova-logo-row">
                <div class="nova-logo-wrap" style="width:{size}px;height:{size}px;">
                    <img class="nova-logo-img" src="data:image/png;base64,{NOVA_LOGO_B64}" />
                </div>
                <div>
                    <div class="nova-wordmark">NOVA</div>
                    <div class="nova-wordmark-sub">{sub_label}</div>
                </div>
            </div>"""
    if clickable:
        inner = f'<a href="?nav=home" target="_self" class="nova-logo-link">{inner}</a>'
    st.markdown(inner, unsafe_allow_html=True)


def fingerprint(model: str, payload: dict) -> str:
    """Stable hash of a model + payload combo, used to detect repeat requests."""
    canonical = json.dumps({"model": model, "payload": payload}, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def kpi_card_v2(icon, label, value, accent, col):
    col.markdown(
        f"""<div class="kpi-card-v2">
                <div class="kpi-accent-bar" style="background:{accent};"></div>
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label-v2">{label}</div>
                <div class="kpi-value-v2">{value}</div>
            </div>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Sidebar — connection & navigation
# --------------------------------------------------------------------------
with st.sidebar:
    nova_logo()
    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">Connection</div>', unsafe_allow_html=True)
    base_url = API_BASE_URL
    st.caption(f"API base URL: `{base_url}`")
    api_key = st.text_input("X-API-Key", type="password", value="")

    ok, msg = api_health(base_url)
    if ok:
        st.success(f"Connected — {msg}")
    else:
        st.error(f"Unreachable — {msg}")

    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow nav-eyebrow-spacer">Navigate</div>', unsafe_allow_html=True)

    for icon, label in NAV_ITEMS:
        is_active = st.session_state.nav_page == label
        if st.button(
            f"{icon}   {label}",
            key=f"nav_{label}",
            width="stretch",
            type="primary" if is_active else "secondary",
        ):
            st.session_state.nav_page = label
            st.rerun()

    page = st.session_state.nav_page

    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.caption(f"Session predictions logged: {len(st.session_state.history)}")
    if st.session_state.history and st.button("🗑️   Clear session history", width="stretch"):
        st.session_state.history = []
        st.rerun()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown('<div class="eyebrow">Bank Customer Churn</div>', unsafe_allow_html=True)
nova_logo(sub_label="", size=64)
st.markdown(
    '<div class="ledger-subtitle">A live dashboard on top of your FastAPI churn-prediction service — '
    'portfolio trends, single-customer scoring, batch scoring, and a record of the modeling work behind it.</div>'
    '<hr class="rule">',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# PAGE: Portfolio Overview
# --------------------------------------------------------------------------
if page == "Portfolio Overview":
    st.markdown('<div class="eyebrow">Reference Portfolio — Churn_Modelling.csv</div>', unsafe_allow_html=True)
    st.caption("Figures below reflect the training dataset used to build these models.")

    c1, c2, c3, c4 = st.columns(4)
    kpi_card_v2("👥", "Total customers", f"{DATASET_STATS['total_customers']:,}", "#6E8CA0", c1)
    kpi_card_v2("📉", "Churn rate", f"{DATASET_STATS['churn_rate']*100:.2f}%", ALERT, c2)
    kpi_card_v2("🧪", "Train / test split", f"{DATASET_STATS['train_size']:,} / {DATASET_STATS['test_size']:,}", TEAL, c3)
    kpi_card_v2("🖊️", "This session's scores", f"{len(st.session_state.history)}", BRASS, c4)

    st.write("")
    left, right = st.columns([1.2, 1], gap="large")

    with left:
        st.markdown("**Where customers come from**")
        geo_items = sorted(DATASET_STATS["geography"].items(), key=lambda x: x[1])
        geo_names = [g for g, _ in geo_items]
        geo_vals = [v for _, v in geo_items]
        total = sum(geo_vals)
        colors_scale = [TEAL, "#6E8CA0", BRASS]
        fig = go.Figure(
            go.Bar(
                x=geo_vals,
                y=geo_names,
                orientation="h",
                marker=dict(color=colors_scale[: len(geo_vals)], cornerradius=8),
                text=[f"{v:,}  ·  {v/total*100:.1f}%" for v in geo_vals],
                textposition="outside",
                textfont=dict(family="IBM Plex Mono", size=13),
            )
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=300,
            xaxis=dict(visible=False, range=[0, max(geo_vals) * 1.28]),
            margin=dict(l=10, r=30, t=10, b=10),
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown("**Retained vs. churned**")
        churned = DATASET_STATS["churn_rate"]
        fig = go.Figure(
            go.Pie(
                labels=["Retained", "Churned"],
                values=[1 - churned, churned],
                hole=0.72,
                marker_colors=[TEAL, ALERT],
                textinfo="none",
                sort=False,
            )
        )
        fig.add_annotation(
            text=f"<b>{churned*100:.1f}%</b><br><span style='font-size:11px;color:{MUTED}'>CHURN RATE</span>",
            showarrow=False, font=dict(family="IBM Plex Mono", size=22, color=TEXT),
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=300, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig, width="stretch")

    st.write("")
    st.markdown("**Customer mix by gender**")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Customers", "Avg. estimated salary"), horizontal_spacing=0.12)
    genders = list(DATASET_STATS["gender"].keys())
    fig.add_trace(
        go.Bar(x=genders, y=list(DATASET_STATS["gender"].values()), marker=dict(color=[TEAL, BRASS], cornerradius=8), showlegend=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=genders, y=list(DATASET_STATS["avg_salary_by_gender"].values()),
            marker=dict(color=[TEAL, BRASS], cornerradius=8), showlegend=False,
        ),
        row=1, col=2,
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=300)
    fig.update_annotations(font=dict(family="Inter", size=13, color=MUTED))
    st.plotly_chart(fig, width="stretch")

    if st.session_state.history:
        st.markdown('<hr class="rule">', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">This Session\'s Scoring Activity</div>', unsafe_allow_html=True)
        hist_df = pd.DataFrame(st.session_state.history)
        fig = go.Figure(
            go.Histogram(
                x=hist_df["probability"] * 100, nbinsx=20,
                marker=dict(color=BRASS, line=dict(color=INK, width=1)),
            )
        )
        avg_prob = hist_df["probability"].mean() * 100
        fig.add_vline(x=avg_prob, line=dict(color=TEAL, width=2, dash="dash"), annotation_text=f"avg {avg_prob:.0f}%", annotation_font_color=TEAL)
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=280, xaxis_title="Predicted churn probability (%)", yaxis_title="Customers scored"
        )
        st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------
# PAGE: Score a Customer
# --------------------------------------------------------------------------
elif page == "Score a Customer":
    st.markdown('<div class="eyebrow">Single-Customer Prediction</div>', unsafe_allow_html=True)
    st.caption("Calls your FastAPI /predict endpoint directly — nothing here is computed locally.")

    form_col, result_col = st.columns([1, 1.1], gap="large")

    with form_col:
        model_choice = st.selectbox("Model", ["Random Forest", "XGBoost"])
        with st.form("predict_form"):
            c1, c2 = st.columns(2)
            with c1:
                credit_score = st.slider("Credit score", 300, 850, 650, help=CUSTOMER_FIELDS_HELP["CreditScore"])
                geography = st.selectbox("Geography", ["France", "Spain", "Germany"], help=CUSTOMER_FIELDS_HELP["Geography"])
                gender = st.selectbox("Gender", ["Male", "Female"], help=CUSTOMER_FIELDS_HELP["Gender"])
                age = st.slider("Age", 18, 100, 38, help=CUSTOMER_FIELDS_HELP["Age"])
                tenure = st.slider("Tenure (years)", 0, 10, 5, help=CUSTOMER_FIELDS_HELP["Tenure"])
            with c2:
                balance = st.number_input("Balance", min_value=0.0, value=75000.0, step=1000.0, help=CUSTOMER_FIELDS_HELP["Balance"])
                num_products = st.slider("Number of products", 1, 4, 1, help=CUSTOMER_FIELDS_HELP["NumOfProducts"])
                has_cr_card = st.selectbox("Has credit card", ["Yes", "No"], help=CUSTOMER_FIELDS_HELP["HasCrCard"])
                is_active = st.selectbox("Active member", ["Yes", "No"], help=CUSTOMER_FIELDS_HELP["IsActiveMember"])
                salary = st.number_input("Estimated salary", min_value=0.0, value=100000.0, step=1000.0, help=CUSTOMER_FIELDS_HELP["EstimatedSalary"])

            submitted = st.form_submit_button("Score this customer", width="stretch")

    if submitted:
        payload = {
            "CreditScore": int(credit_score),
            "Geography": geography,
            "Gender": gender,
            "Age": int(age),
            "Tenure": int(tenure),
            "Balance": float(balance),
            "NumOfProducts": int(num_products),
            "HasCrCard": 1 if has_cr_card == "Yes" else 0,
            "IsActiveMember": 1 if is_active == "Yes" else 0,
            "EstimatedSalary": float(salary),
        }

        req_fp = fingerprint(model_choice, payload)
        is_duplicate = st.session_state.get("last_predict_fp") == req_fp

        if is_duplicate:
            st.info("⚠️ Same customer and model as your last request — reusing the previous result instead of calling the API again.")
            success, result = True, st.session_state["last_predict_result"]
        else:
            with st.spinner("Calling the API..."):
                success, result = call_predict(base_url, api_key, model_choice, payload)
            if success:
                st.session_state["last_predict_fp"] = req_fp
                st.session_state["last_predict_result"] = result

        with result_col:
            if not success:
                st.error(f"Prediction failed: {result}")
            else:
                prob = result["Churn_Probability"]
                pred = result["Churn_Prediction"]
                band, badge_class, _ = risk_band(prob)

                st.plotly_chart(gauge_chart(prob), width="stretch")
                st.markdown(
                    f'<span class="badge {badge_class}">{band.upper()} RISK</span>&nbsp;&nbsp;'
                    f'<span style="color:{MUTED};">Predicted label: '
                    f'<b style="color:{TEXT};">{"Will churn" if pred else "Will stay"}</b></span>',
                    unsafe_allow_html=True,
                )
                st.write("")
                st.markdown("**Where this customer sits on the model's top drivers**")
                top_feats = dict(sorted(FEATURE_IMPORTANCE.items(), key=lambda x: -x[1])[:5])
                fig = go.Figure(go.Bar(x=list(top_feats.values()), y=list(top_feats.keys()), orientation="h", marker_color=BRASS))
                fig.update_layout(template=PLOTLY_TEMPLATE, height=260, xaxis_title="Relative importance (Random Forest)")
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, width="stretch")

                if not is_duplicate:
                    st.session_state.history.append(
                        {
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "model": model_choice,
                            "probability": prob,
                            "prediction": "Churn" if pred else "Stay",
                            **payload,
                        }
                    )

    if st.session_state.history:
        st.markdown('<hr class="rule">', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">Recent Scores (this session)</div>', unsafe_allow_html=True)
        recent = pd.DataFrame(st.session_state.history[::-1][:10])
        recent_display = recent[["timestamp", "model", "prediction", "probability"]].copy()
        recent_display["probability"] = recent_display["probability"].apply(lambda p: f"{p:.1%}")
        st.dataframe(recent_display, width="stretch", hide_index=True)

# --------------------------------------------------------------------------
# PAGE: Batch Scoring
# --------------------------------------------------------------------------
elif page == "Batch Scoring":
    st.markdown('<div class="eyebrow">Batch Scoring</div>', unsafe_allow_html=True)
    st.caption(
        "Upload a CSV with columns: "
        + ", ".join(CUSTOMER_FIELDS_HELP.keys())
        + ". Each row is sent to the API individually."
    )

    model_choice_b = st.selectbox("Model", ["Random Forest", "XGBoost"], key="batch_model")
    uploaded = st.file_uploader("Customer CSV", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = None

        if df is not None:
            missing = [c for c in CUSTOMER_FIELDS_HELP if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                st.dataframe(df.head(), width="stretch")
                if st.button(f"Score {len(df)} customers", width="stretch"):
                    batch_fp = fingerprint(model_choice_b, {"csv_hash": hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()})
                    is_duplicate_batch = st.session_state.get("last_batch_fp") == batch_fp

                    if is_duplicate_batch:
                        st.info("⚠️ This is the same file and model as your last batch — reusing the previous results instead of calling the API again.")
                        res_df = st.session_state["last_batch_result"]
                    else:
                        results = []
                        progress = st.progress(0.0, text="Scoring...")
                        for i, row in df.iterrows():
                            payload = {k: row[k] for k in CUSTOMER_FIELDS_HELP}
                            # Coerce numeric types the API expects
                            for k in ["CreditScore", "Age", "Tenure", "NumOfProducts", "HasCrCard", "IsActiveMember"]:
                                payload[k] = int(payload[k])
                            for k in ["Balance", "EstimatedSalary"]:
                                payload[k] = float(payload[k])

                            success, result = call_predict(base_url, api_key, model_choice_b, payload)
                            if success:
                                results.append({**payload, "Churn_Prediction": result["Churn_Prediction"], "Churn_Probability": result["Churn_Probability"]})
                            else:
                                results.append({**payload, "Churn_Prediction": None, "Churn_Probability": None, "error": result})
                            progress.progress((i + 1) / len(df), text=f"Scoring... {i+1}/{len(df)}")
                        progress.empty()

                        res_df = pd.DataFrame(results)
                        st.session_state["last_batch_fp"] = batch_fp
                        st.session_state["last_batch_result"] = res_df

                    st.success(f"Scored {len(res_df)} customers.")

                    c1, c2 = st.columns(2)
                    valid = res_df.dropna(subset=["Churn_Probability"])
                    with c1:
                        st.markdown("**Risk distribution**")
                        fig = go.Figure(go.Histogram(x=valid["Churn_Probability"] * 100, nbinsx=20, marker_color=TEAL))
                        fig.update_layout(template=PLOTLY_TEMPLATE, height=300, xaxis_title="Predicted churn probability (%)")
                        st.plotly_chart(fig, width="stretch")
                    with c2:
                        st.markdown("**Predicted outcome split**")
                        counts = valid["Churn_Prediction"].value_counts()
                        fig = go.Figure(
                            go.Pie(
                                labels=["Will stay" if not k else "Will churn" for k in counts.index],
                                values=counts.values,
                                marker_colors=[TEAL, ALERT],
                                hole=0.55,
                            )
                        )
                        fig.update_layout(template=PLOTLY_TEMPLATE, height=300, showlegend=True)
                        st.plotly_chart(fig, width="stretch")

                    st.markdown("**Full results**")
                    st.dataframe(res_df, width="stretch", hide_index=True)
                    st.download_button(
                        "Download scored CSV",
                        res_df.to_csv(index=False).encode("utf-8"),
                        file_name="churn_scores.csv",
                        mime="text/csv",
                        width="stretch",
                    )

# --------------------------------------------------------------------------
# PAGE: Model Insights
# --------------------------------------------------------------------------
elif page == "Model Insights":
    st.markdown('<div class="eyebrow">Model Development Journey</div>', unsafe_allow_html=True)
    st.caption("Every configuration tried, in the order it was tested — test-set F1 climbs as imbalance handling and tuning are added.")

    ledger_df = pd.DataFrame(MODEL_LEDGER)
    ledger_df["step"] = range(1, len(ledger_df) + 1)
    ledger_df["short"] = ledger_df["entry"].str.replace("Random Forest", "RF").str.replace("Logistic Regression", "Logistic")
    best_idx = ledger_df["test_f1"].idxmax()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ledger_df["step"], y=ledger_df["train_f1"], name="Train F1",
            mode="lines+markers", line=dict(color="#6E8CA0", width=2, dash="dot"),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ledger_df["step"], y=ledger_df["test_f1"], name="Test F1",
            mode="lines+markers", line=dict(color=BRASS, width=3), marker=dict(size=8),
            fill="tozeroy", fillcolor="rgba(201,162,39,0.08)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[ledger_df.loc[best_idx, "step"]], y=[ledger_df.loc[best_idx, "test_f1"]],
            mode="markers+text", marker=dict(size=16, color=TEAL, line=dict(color=INK, width=2)),
            text=["★ best"], textposition="top center", textfont=dict(color=TEAL, size=12),
            showlegend=False,
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=380,
        xaxis=dict(tickmode="array", tickvals=list(ledger_df["step"]), ticktext=list(ledger_df["short"]), tickangle=-25),
        yaxis_title="F1 score", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown('<div class="eyebrow">Leaderboard — Ranked by Test F1</div>', unsafe_allow_html=True)
        st.write("")
        ranked = ledger_df.sort_values("test_f1", ascending=False).reset_index(drop=True)
        max_f1 = ranked["test_f1"].max()
        rows_html = ""
        for i, row in ranked.iterrows():
            rank = i + 1
            pct = row["test_f1"] / max_f1 * 100
            color = BRASS if rank == 1 else (TEAL if rank <= 3 else "#6E8CA0")
            num_class = "gold" if rank == 1 else ""
            rows_html += (
                f'<div class="rank-row">'
                f'<div class="rank-num {num_class}">{rank}</div>'
                f'<div class="rank-label">{row["entry"]}<br><span style="color:{MUTED};font-size:0.75rem;">{row["note"]}</span></div>'
                f'<div class="rank-bar-track"><div class="rank-bar-fill" style="width:{pct:.0f}%;background:{color};"></div></div>'
                f'<div class="rank-score">{row["test_f1"]:.3f}</div>'
                f'</div>'
            )
        st.markdown(rows_html, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="eyebrow">What Drives the Random Forest Model</div>', unsafe_allow_html=True)
        st.write("")
        feat_df = pd.DataFrame(
            {"Feature": list(FEATURE_IMPORTANCE.keys()), "Importance": list(FEATURE_IMPORTANCE.values())}
        ).sort_values("Importance", ascending=True)
        top5_cut = feat_df["Importance"].nlargest(5).min()
        colors = [BRASS if v >= top5_cut else "rgba(63,167,150,0.55)" for v in feat_df["Importance"]]
        fig = go.Figure(
            go.Bar(
                x=feat_df["Importance"], y=feat_df["Feature"], orientation="h",
                marker=dict(color=colors, cornerradius=6),
                text=[f"{v*100:.1f}%" for v in feat_df["Importance"]], textposition="outside",
                textfont=dict(family="IBM Plex Mono", size=11),
            )
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=420, xaxis=dict(visible=False, range=[0, feat_df["Importance"].max() * 1.25]))
        st.plotly_chart(fig, width="stretch")

    with st.container(border=True):
        st.markdown('<div class="eyebrow">Notes</div>', unsafe_allow_html=True)
        st.markdown(
            "- Both deployed models (`forest_tuned.pkl`, `xgb-tuned.pkl`) were selected after this comparison, "
            "prioritizing test-set **F1 score** because churn is a minority class (≈20% of customers).\n"
            "- Class imbalance was addressed two ways during experimentation: **class weighting** and **SMOTE** oversampling; "
            "final Random Forest and XGBoost picks use class weighting and hyperparameter tuning.\n"
            "- `Age` and `NumOfProducts` (highlighted in brass) are by far the strongest predictors of churn in the "
            "Random Forest model — together they account for over 60% of its decision weight."
        )

# --------------------------------------------------------------------------
# PAGE: Usage Analytics
# --------------------------------------------------------------------------
elif page == "Usage Analytics":
    st.markdown('<div class="eyebrow">Who\'s Using the Models</div>', unsafe_allow_html=True)
    st.caption("Live counts pulled from Supabase — every successful /predict call anywhere (this dashboard or direct API calls) is logged and counted.")

    if not api_key:
        st.warning("Enter your X-API-Key in the sidebar to load usage stats.")
    else:
        success, stats = get_usage_stats(base_url, api_key)
        if not success:
            st.error(f"Could not load usage stats: {stats}")
        else:
            forest_count = stats.get("forest", {}).get("count", 0)
            xgb_count = stats.get("xgboost", {}).get("count", 0)
            total = forest_count + xgb_count

            c1, c2, c3 = st.columns(3)
            kpi_card_v2("🌲", "Random Forest calls", f"{forest_count:,}", TEAL, c1)
            kpi_card_v2("⚡", "XGBoost calls", f"{xgb_count:,}", BRASS, c2)
            kpi_card_v2("Σ", "Total predictions served", f"{total:,}", "#6E8CA0", c3)

            st.write("")
            left, right = st.columns([1, 1.3], gap="large")

            with left:
                st.markdown("**Calls by model**")
                fig = go.Figure(
                    go.Bar(
                        x=["Random Forest", "XGBoost"], y=[forest_count, xgb_count],
                        marker=dict(color=[TEAL, BRASS], cornerradius=8),
                        text=[forest_count, xgb_count], textposition="outside",
                        textfont=dict(family="IBM Plex Mono", size=13),
                    )
                )
                fig.update_layout(template=PLOTLY_TEMPLATE, height=320, yaxis_title="Predictions served")
                st.plotly_chart(fig, width="stretch")

            with right:
                st.markdown("**Requests over time**")
                all_ts = []
                for model_name in ("forest", "xgboost"):
                    for ts in stats.get(model_name, {}).get("timestamps", []):
                        all_ts.append({"timestamp": ts, "model": model_name})
                if not all_ts:
                    st.info("No requests logged yet — once the API gets some traffic, this chart fills in.")
                else:
                    ts_df = pd.DataFrame(all_ts)
                    ts_df["timestamp"] = pd.to_datetime(ts_df["timestamp"])

                    span = ts_df["timestamp"].max() - ts_df["timestamp"].min()
                    if span <= pd.Timedelta(hours=3):
                        freq, tick_fmt, x_title = "5min", "%H:%M", "Time (5-minute buckets)"
                    elif span <= pd.Timedelta(days=3):
                        freq, tick_fmt, x_title = "h", "%b %d, %H:%M", "Time (hourly)"
                    elif span <= pd.Timedelta(days=90):
                        freq, tick_fmt, x_title = "D", "%b %d", "Time (daily)"
                    else:
                        freq, tick_fmt, x_title = "W", "%b %d", "Time (weekly)"

                    bucketed = (
                        ts_df.set_index("timestamp")
                        .resample(freq)
                        .size()
                        .rename("calls")
                        .reset_index()
                    )
                    # Pad a bit of empty space either side so a single point isn't stranded mid-axis
                    freq_as_timedelta = pd.Timedelta(freq if freq[0].isdigit() else f"1{freq}")
                    pad = max((bucketed["timestamp"].max() - bucketed["timestamp"].min()) * 0.05, freq_as_timedelta)
                    x_range = [bucketed["timestamp"].min() - pad, bucketed["timestamp"].max() + pad]

                    fig = go.Figure(
                        go.Scatter(
                            x=bucketed["timestamp"], y=bucketed["calls"],
                            mode="lines+markers",
                            line=dict(color=TEAL, width=2.5, shape="spline" if len(bucketed) > 2 else "linear"),
                            marker=dict(size=7, color=BRASS, line=dict(color=INK, width=1)),
                            fill="tozeroy",
                            fillcolor="rgba(63,167,150,0.22)",
                            hovertemplate=f"%{{x|{tick_fmt}}}<br><b>%{{y}} calls</b><extra></extra>",
                        )
                    )
                    fig.update_layout(
                        template=PLOTLY_TEMPLATE, height=320,
                        yaxis_title="Predictions", xaxis_title=x_title,
                        hovermode="x unified",
                        yaxis=dict(rangemode="tozero"),
                        xaxis=dict(range=x_range, tickformat=tick_fmt),
                    )
                    st.plotly_chart(fig, width="stretch")
                    st.caption(f"{len(ts_df)} request(s) logged between {ts_df['timestamp'].min():%b %d, %Y %H:%M} and {ts_df['timestamp'].max():%b %d, %Y %H:%M}.")

            st.caption(
                "Requests are logged to Supabase on every successful prediction, so this chart reflects the full "
                "history of API traffic — not just what's happened since the last deployment."
            )


# --------------------------------------------------------------------------
st.markdown('<hr class="rule">', unsafe_allow_html=True)
st.caption("NOVA · a Streamlit front-end for the Churn-Detection FastAPI service · predictions are never computed locally.")