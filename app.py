import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------
# Streamlit config
# -----------------------
st.set_page_config(
    page_title="Digital Empathy Engine",
    layout="wide"
)

# -----------------------
# FILE PATHS (ROOT LEVEL)
# -----------------------
DATA_PATH = "sample_digital_exhaust.csv"
BASELINES_PATH = "cultural_baselines.csv"
LR_PATH = "logreg_explain_model.pkl"
RF_PATH = "rf_risk_model.pkl"

FEATURES = [
    "norm_after_hours",
    "norm_reply",
    "norm_meetings",
    "collab_index",
    "weekend_load"
]

# -----------------------
# FEATURE ENGINEERING
# -----------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in [
        "emails_sent",
        "emails_after_hours",
        "avg_reply_minutes",
        "meetings_minutes",
        "unique_collaborators",
        "weekend_activity_minutes"
    ]:
        df[col] = df[col].clip(lower=0)

    df["after_hours_pct"] = (
        df["emails_after_hours"] /
        df["emails_sent"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)

    df["meeting_load"] = df["meetings_minutes"]
    df["response_delay"] = df["avg_reply_minutes"]
    df["collab_index"] = df["unique_collaborators"]
    df["weekend_load"] = df["weekend_activity_minutes"]

    return df


def apply_cultural_normalization(df: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(base, on="country", how="left")

    out["baseline_after_hours_pct"] = out["baseline_after_hours_pct"].replace(0, np.nan)
    out["baseline_reply_minutes"] = out["baseline_reply_minutes"].replace(0, np.nan)
    out["baseline_meeting_minutes"] = out["baseline_meeting_minutes"].replace(0, np.nan)

    out["baseline_after_hours_pct"].fillna(
        base["baseline_after_hours_pct"].median(), inplace=True
    )
    out["baseline_reply_minutes"].fillna(
        base["baseline_reply_minutes"].median(), inplace=True
    )
    out["baseline_meeting_minutes"].fillna(
        base["baseline_meeting_minutes"].median(), inplace=True
    )

    out["norm_after_hours"] = out["after_hours_pct"] / out["baseline_after_hours_pct"]
    out["norm_reply"] = out["response_delay"] / out["baseline_reply_minutes"]
    out["norm_meetings"] = out["meeting_load"] / out["baseline_meeting_minutes"]

    for c in ["norm_after_hours", "norm_reply", "norm_meetings"]:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(1).clip(0, 5)

    return out


def risk_band(score):
    if score < 30:
        return "Low"
    elif score < 60:
        return "Medium"
    return "High"


def dominant_driver(row):
    scores = {
        "Meeting Overload": row["norm_meetings"],
        "After-hours Work": row["norm_after_hours"],
        "Weekend Work": row["weekend_load"],
        "Response Delay": row["norm_reply"],
        "Low Collaboration": -row["collab_index"]
    }
    return max(scores, key=scores.get)


def jitai_recommendations(row):
    emp, mgr, hr = [], [], []

    if row["risk_band"] == "Low":
        emp.append("Healthy work pattern detected. Maintain boundaries and recovery time.")
        mgr.append("No action required. Monitor at team level.")
        hr.append("No action required.")
        return emp, mgr, hr

    if row["risk_band"] == "Medium":
        emp.append("Slight workload elevation detected. Reduce after-hours work.")
        mgr.append("Review meeting load and task distribution.")
        hr.append("Monitor trends; no escalation needed yet.")

    if row["risk_band"] == "High":
        emp.append("High burnout risk detected. Prioritize recovery and focus blocks.")
        mgr.append("Schedule a confidential workload support check-in.")
        hr.append("Consider confidential HR review if risk persists.")

    if row["norm_meetings"] >= 1.3:
        emp.append("High meeting load detected. Decline non-essential meetings.")
        mgr.append("Reduce recurring meetings; enforce agendas.")

    if row["norm_after_hours"] >= 1.3:
        emp.append("After-hours work is high. Shift tasks into work hours.")
        mgr.append("Clarify response-time expectations.")

    if row["weekend_load"] >= 60:
        emp.append("Weekend work detected. Ensure adequate rest.")
        mgr.append("Avoid weekend workload spillover.")

    if row["collab_index"] <= 3:
        emp.append("Low collaboration detected. Engage with peers.")
        mgr.append("Encourage buddy system or team syncs.")

    return emp, mgr, hr


# -----------------------
# UI START
# -----------------------
st.title("🧠 Digital Empathy Engine")
st.caption("Dual-model system: Random Forest (risk scoring) + Logistic Regression (explainability)")

st.sidebar.header("Dataset")

uploaded = st.sidebar.file_uploader(
    "Upload Digital Exhaust CSV (optional)",
    type=["csv"]
)

# Load data
if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
else:
    if not os.path.exists(DATA_PATH):
        st.error("sample_digital_exhaust.csv not found in repository.")
        st.stop()
    df_raw = pd.read_csv(DATA_PATH)

if not os.path.exists(BASELINES_PATH):
    st.error("cultural_baselines.csv not found in repository.")
    st.stop()

base = pd.read_csv(BASELINES_PATH)

# Load models
if not (os.path.exists(LR_PATH) and os.path.exists(RF_PATH)):
    st.error("Model files not found. Ensure .pkl files are in repo root.")
    st.stop()

lr_model = joblib.load(LR_PATH)
rf_model = joblib.load(RF_PATH)

# Feature pipeline
df = apply_cultural_normalization(add_features(df_raw), base)

X = df[FEATURES]

df["burnout_risk_score"] = (rf_model.predict_proba(X)[:, 1] * 100).round(1)
df["risk_band"] = df["burnout_risk_score"].apply(risk_band)
df["dominant_driver"] = df.apply(dominant_driver, axis=1)

# -----------------------
# TABS
# -----------------------
tab1, tab2 = st.tabs(["📊 Overview (HR)", "👤 Employee Drilldown"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employees", df["employee_id"].nunique())
    c2.metric("Avg Risk Score", round(df["burnout_risk_score"].mean(), 1))
    c3.metric("High Risk Cases", int((df["risk_band"] == "High").sum()))
    c4.metric("Records", len(df))

    fig = px.histogram(
        df,
        x="burnout_risk_score",
        nbins=25,
        title="Burnout Risk Score Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top High-Risk Records")
    st.dataframe(
        df.sort_values("burnout_risk_score", ascending=False)
        .head(10)[["employee_id", "country", "date", "burnout_risk_score", "risk_band"]],
        use_container_width=True
    )

with tab2:
    emp_id = st.selectbox(
        "Select Employee",
        sorted(df["employee_id"].astype(str).unique())
    )

    emp_df = df[df["employee_id"].astype(str) == emp_id].sort_values("date")
    latest = emp_df.iloc[-1]

    a, b, c = st.columns(3)
    a.metric("Risk Score", latest["burnout_risk_score"])
    b.metric("Risk Band", latest["risk_band"])
    c.metric("Dominant Driver", latest["dominant_driver"])

    trend = px.line(
        emp_df,
        x="date",
        y="burnout_risk_score",
        title="Burnout Risk Trend"
    )
    st.plotly_chart(trend, use_container_width=True)

    st.subheader("🎯 JITAI Recommendations")
    emp_msgs, mgr_msgs, hr_msgs = jitai_recommendations(latest)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Employee**")
        for m in emp_msgs:
            st.write("- " + m)

    with col2:
        st.markdown("**Manager**")
        for m in mgr_msgs:
            st.write("- " + m)

    with col3:
        st.markdown("**HR**")
        for m in hr_msgs:
            st.write("- " + m)

st.caption(
    "Decision-support only. Use with consent, privacy protection, and ethical HR governance."
)
