import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Digital Empathy Engine", layout="wide")

# Paths
DATA_PATH = "data/sample_digital_exhaust.csv"
BASELINES_PATH = "data/cultural_baselines.csv"
LR_PATH = "models/logreg_explain_model.pkl"
RF_PATH = "models/rf_risk_model.pkl"

FEATURES = ["norm_after_hours", "norm_reply", "norm_meetings", "collab_index", "weekend_load"]

# -----------------------
# Feature engineering
# -----------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["emails_sent"] = df["emails_sent"].clip(lower=0)
    df["emails_after_hours"] = df["emails_after_hours"].clip(lower=0)
    df["avg_reply_minutes"] = df["avg_reply_minutes"].clip(lower=0)
    df["meetings_minutes"] = df["meetings_minutes"].clip(lower=0)
    df["unique_collaborators"] = df["unique_collaborators"].clip(lower=0)
    df["weekend_activity_minutes"] = df["weekend_activity_minutes"].clip(lower=0)

    df["after_hours_pct"] = df["emails_after_hours"] / df["emails_sent"].replace(0, np.nan)
    df["after_hours_pct"] = df["after_hours_pct"].fillna(0.0).clip(0, 1)

    df["meeting_load"] = df["meetings_minutes"]
    df["response_delay"] = df["avg_reply_minutes"]
    df["collab_index"] = df["unique_collaborators"]
    df["weekend_load"] = df["weekend_activity_minutes"]
    return df

def apply_cultural_normalization(df: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    base = base.copy()

    base["baseline_after_hours_pct"] = base["baseline_after_hours_pct"].replace(0, np.nan)
    base["baseline_reply_minutes"] = base["baseline_reply_minutes"].replace(0, np.nan)
    base["baseline_meeting_minutes"] = base["baseline_meeting_minutes"].replace(0, np.nan)

    out = df.merge(base, on="country", how="left")

    out["baseline_after_hours_pct"] = out["baseline_after_hours_pct"].fillna(base["baseline_after_hours_pct"].median())
    out["baseline_reply_minutes"] = out["baseline_reply_minutes"].fillna(base["baseline_reply_minutes"].median())
    out["baseline_meeting_minutes"] = out["baseline_meeting_minutes"].fillna(base["baseline_meeting_minutes"].median())

    out["norm_after_hours"] = out["after_hours_pct"] / out["baseline_after_hours_pct"]
    out["norm_reply"] = out["response_delay"] / out["baseline_reply_minutes"]
    out["norm_meetings"] = out["meeting_load"] / out["baseline_meeting_minutes"]

    for c in ["norm_after_hours", "norm_reply", "norm_meetings"]:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0, 5)

    return out

def risk_band(score):
    if score < 30:
        return "Low"
    elif score < 60:
        return "Medium"
    return "High"

def driver_flags(row):
    return {
        "Meeting Overload": row["norm_meetings"] >= 1.3,
        "After-hours Work": row["norm_after_hours"] >= 1.3,
        "Weekend Work": row["weekend_load"] >= 60,
        "Response Delay": row["norm_reply"] >= 1.4,
        "Low Collaboration": row["collab_index"] <= 3,
    }

def dominant_driver(row):
    scores = {
        "Meeting Overload": row["norm_meetings"],
        "After-hours Work": row["norm_after_hours"],
        "Weekend Work": row["weekend_load"],
        "Response Delay": row["norm_reply"],
        "Low Collaboration": -row["collab_index"],
    }
    return max(scores, key=scores.get)

def jitai_recommendations(row):
    band = row["risk_band"]
    flags = driver_flags(row)

    employee_msgs, manager_msgs, hr_msgs = [], [], []

    if band == "Low":
        employee_msgs.append("You’re maintaining a healthy work rhythm. Keep boundaries and take regular breaks.")
        manager_msgs.append("No action required. Continue monitoring workload balance at team level.")
        hr_msgs.append("No action required. Monitor aggregated trends only.")
        return employee_msgs, manager_msgs, hr_msgs

    if band == "Medium":
        employee_msgs.append("Workload signals are slightly elevated. Add focus blocks and reduce after-hours tasks.")
        manager_msgs.append("Check workload distribution and meeting load; encourage async communication.")
        hr_msgs.append("Monitor; no escalation unless sustained for multiple days/weeks.")

    if band == "High":
        employee_msgs.append("High strain detected. Consider a short recovery plan (focus blocks + reduced after-hours work).")
        manager_msgs.append("Schedule a confidential support check-in; reduce meetings and rebalance tasks.")
        hr_msgs.append("Confidential HR review recommended if high risk persists (privacy-safe summary only).")

    if flags["Meeting Overload"]:
        employee_msgs.append("Meeting load is high. Decline optional meetings and request agendas/outcomes.")
        manager_msgs.append("Reduce recurring meetings; enforce agenda-first and invite-only policy.")
        hr_msgs.append("Team-level meeting hygiene guidelines and manager training may help.")

    if flags["After-hours Work"]:
        employee_msgs.append("After-hours activity is elevated. Shift non-urgent tasks to work hours.")
        manager_msgs.append("Reinforce boundary norms and clarify response-time expectations.")
        hr_msgs.append("Consider policy reminders on boundary-respecting practices.")

    if flags["Weekend Work"]:
        employee_msgs.append("Weekend work is high. Prioritize recovery time unless critical.")
        manager_msgs.append("Ensure weekend work is not expected; plan capacity to avoid spillover.")
        hr_msgs.append("Track weekend work as a wellbeing risk indicator at team level.")

    if flags["Response Delay"]:
        employee_msgs.append("Response delays may reflect overload. Reduce context switching and prioritize tasks.")
        manager_msgs.append("Check blockers and task load; reassign urgent items if needed.")
        hr_msgs.append("Monitor for staffing/process constraints if delays persist.")

    if flags["Low Collaboration"]:
        employee_msgs.append("Low collaboration detected. Reach out to peers or join a team sync for support.")
        manager_msgs.append("Encourage peer buddy system and regular 1:1s.")
        hr_msgs.append("Monitor isolation risk, especially combined with high workload.")

    return employee_msgs, manager_msgs, hr_msgs

# -----------------------
# Load inputs
# -----------------------
st.title("🧠 Digital Empathy Engine")
st.caption("Dual-model approach: Random Forest for scoring + Logistic Regression for explainability")

st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload Digital Exhaust CSV (optional)", type=["csv"])

df_raw = pd.read_csv(uploaded) if uploaded is not None else pd.read_csv(DATA_PATH)
base = pd.read_csv(BASELINES_PATH)

# Load models
if not (os.path.exists(LR_PATH) and os.path.exists(RF_PATH)):
    st.error("Missing model files. Place them in /models: logreg_explain_model.pkl and rf_risk_model.pkl")
    st.stop()

lr_model = joblib.load(LR_PATH)
rf_model = joblib.load(RF_PATH)

df = apply_cultural_normalization(add_features(df_raw), base)

# RF = risk score (primary)
X = df[FEATURES]
df["burnout_risk_score"] = (rf_model.predict_proba(X)[:, 1] * 100).round(1)
df["risk_band"] = df["burnout_risk_score"].apply(risk_band)
df["dominant_driver"] = df.apply(dominant_driver, axis=1)

# LR = explainability (coefficients are global)
lr_coef = lr_model.named_steps["clf"].coef_[0]
coef_df = pd.DataFrame({"Feature": FEATURES, "Coefficient": lr_coef}).sort_values("Coefficient", ascending=False)

# -----------------------
# Tabs
# -----------------------
tab1, tab2 = st.tabs(["📊 Overview (HR)", "👤 Employee Drilldown"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employees", df["employee_id"].nunique())
    c2.metric("Avg Risk Score", float(df["burnout_risk_score"].mean().round(1)))
    c3.metric("High Risk Cases", int((df["risk_band"] == "High").sum()))
    c4.metric("Data Points", len(df))

    left, right = st.columns([2, 1])
    with left:
        fig = px.histogram(df, x="burnout_risk_score", nbins=25, title="Risk Score Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.write("### 🔥 Top 10 Risk Records")
        top = df.sort_values("burnout_risk_score", ascending=False).head(10)
        st.dataframe(top[["employee_id","country","date","burnout_risk_score","risk_band"]], use_container_width=True)

    st.write("### 🌍 Country-wise Risk Summary")
    country_summary = df.groupby("country").agg(
        avg_risk=("burnout_risk_score","mean"),
        high_count=("risk_band", lambda x: (x=="High").sum()),
        records=("risk_band","count")
    ).reset_index()
    country_summary["avg_risk"] = country_summary["avg_risk"].round(1)
    st.dataframe(country_summary, use_container_width=True)

    st.write("### 🧾 Explainability (Logistic Regression Coefficients)")
    st.dataframe(coef_df, use_container_width=True)

with tab2:
    emp_ids = sorted(df["employee_id"].astype(str).unique())
    selected_emp = st.selectbox("Select Employee", emp_ids)

    emp_df = df[df["employee_id"].astype(str) == selected_emp].sort_values("date")
    latest = emp_df.iloc[-1]

    a, b, c = st.columns(3)
    a.metric("Latest Risk Score (RF)", latest["burnout_risk_score"])
    b.metric("Risk Band", latest["risk_band"])
    c.metric("Dominant Driver", latest["dominant_driver"])

    trend = px.line(emp_df, x="date", y="burnout_risk_score", title="Risk Trend Over Time (RF Score)")
    st.plotly_chart(trend, use_container_width=True)

    st.write("### 🔎 Latest Driver Values")
    drivers = pd.DataFrame({
        "Driver": ["norm_meetings","norm_after_hours","norm_reply","weekend_load","collab_index"],
        "Value": [
            float(latest["norm_meetings"]),
            float(latest["norm_after_hours"]),
            float(latest["norm_reply"]),
            float(latest["weekend_load"]),
            float(latest["collab_index"])
        ]
    })
    st.dataframe(drivers, use_container_width=True)

    st.write("### 🎯 JITAI Recommendations (Latest)")
    emp_msgs, mgr_msgs, hr_msgs = jitai_recommendations(latest)
    colE, colM, colH = st.columns(3)

    with colE:
        st.subheader("Employee")
        for m in emp_msgs:
            st.write(f"- {m}")

    with colM:
        st.subheader("Manager")
        for m in mgr_msgs:
            st.write(f"- {m}")

    with colH:
        st.subheader("HR")
        for m in hr_msgs:
            st.write(f"- {m}")

st.caption("Decision-support only. Use with consent, privacy safeguards, and fairness checks.")
