# --- Imports ---
import os
import io
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime, timedelta

# --- Load environment ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.sidebar.error("Set GOOGLE_API_KEY in .env before running the app.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- Helper: Load call log file ---
def load_call_logs(file_bytes, filename):
    name = filename.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif name.endswith(".json"):
            df = pd.read_json(io.BytesIO(file_bytes))
        elif name.endswith((".txt", ".log")):
            text = file_bytes.decode(errors="ignore")
            df = pd.read_csv(io.StringIO(text), sep=",")
        else:
            st.error("Unsupported file format")
            return None
        df.columns = df.columns.str.lower()
        return df
    except Exception as e:
        st.warning(f"Could not parse file: {e}")
        return None

# --- Geo lookup by phone prefix (mocked) ---
def get_location_from_number(number):
    prefix = str(number)[:3]
    region_map = {
        "982": ("Mumbai", "India"),
        "981": ("Delhi", "India"),
        "984": ("Chennai", "India"),
        "987": ("Pune", "India"),
        "880": ("Bangladesh", "International"),
    }
    return region_map.get(prefix, ("Unknown", "Unknown"))

# --- AI prompt builder ---
def build_ai_prompt(filename, df):
    prompt = [
        f"Analyze the telecom call log data from {filename}.",
        "Identify suspicious callers or numbers showing spam-like behavior.",
        "Explain the reasons, risks, and possible legal or security implications.",
        "Suggest recommendations for the telecom operator to mitigate misuse.",
        "\nHere are sample call records:",
        df.head(10).to_csv(index=False)
    ]
    return "\n".join(prompt)

def call_gemini(prompt_text, model="gemini-2.5-flash"):
    try:
        model = genai.GenerativeModel(model)
        response = model.generate_content(prompt_text)
        return getattr(response, "text", str(response))
    except Exception as e:
        return f"ERROR calling Gemini: {e}"

# --- Streamlit UI ---
st.set_page_config(page_title="CallInsightAI", layout="wide")
st.title("📞 CallInsightAI — Telecom Call Log Analyzer")
st.write("Upload telecom call logs (CSV/JSON/TXT) to analyze caller behavior and detect suspicious patterns using Gemini AI.")

uploaded = st.file_uploader("Upload Call Log", type=["csv", "json", "txt", "log"])

if uploaded:
    df = load_call_logs(uploaded.read(), uploaded.name)
    if df is not None:
        st.success(f"✅ Loaded {len(df)} records successfully.")

        # ---- Auto-map columns to expected names ----
        if 'phone number' in df.columns:
            df.rename(columns={'phone number': 'caller'}, inplace=True)

        if 'duration' not in df.columns:
            duration_cols = ['day mins', 'eve mins', 'night mins', 'intl mins']
            if all(col in df.columns for col in duration_cols):
                df['duration'] = df[duration_cols].sum(axis=1) * 60  # mins → sec

        if 'timestamp' not in df.columns:
            start = pd.Timestamp('2025-01-01 08:00')
            df['timestamp'] = [start + timedelta(minutes=i*5) for i in range(len(df))]

        # ---- Custom CSS for buttons ----
        st.markdown("""
        <style>
        .button-container {
            display: flex;
            justify-content: center;
            gap: 25px;
            margin-bottom: 25px;
        }
        .stButton>button {
            width: 210px;
            height: 60px;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            color: white;
            background-color: #0056b3;
            border: none;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #007bff;
            transform: scale(1.04);
        }
        .stButton>button:focus {
            outline: none;
            background-color: #003f7f;
        }
        </style>
        """, unsafe_allow_html=True)

        # ---- Button Row ----
        st.markdown("<div class='button-container'>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            pattern_btn = st.button("📈 Call Pattern Analysis")
        with col2:
            suspicious_btn = st.button("🧩 Suspicious Caller Detection")
        with col3:
            crossregion_btn = st.button("🌐 Cross-Region Call Analysis")
        with col4:
            ai_btn = st.button("🧠 AI Risk Analysis")
        st.markdown("</div>", unsafe_allow_html=True)

        selected_mode = None
        if pattern_btn:
            selected_mode = "pattern"
        elif suspicious_btn:
            selected_mode = "suspicious"
        elif crossregion_btn:
            selected_mode = "crossregion"
        elif ai_btn:
            selected_mode = "ai"

        # ---- Call Pattern Analysis ----
        if selected_mode == "pattern":
            st.subheader("📈 Call Pattern Analysis")
            if "timestamp" in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df['hour'] = df['timestamp'].dt.hour
                calls_per_hour = df.groupby('hour').size().reset_index(name='calls')
                fig_hour = px.bar(calls_per_hour, x='hour', y='calls', title="Calls per Hour")
                st.plotly_chart(fig_hour, use_container_width=True)
            if "duration" in df.columns:
                st.metric("Average Call Duration", f"{df['duration'].mean():.2f} sec")
            if "caller" in df.columns:
                top_callers = df["caller"].value_counts().nlargest(10).reset_index()
                top_callers.columns = ['Caller', 'Call Count']
                fig_top = px.bar(top_callers, x='Caller', y='Call Count', title="Top 10 Frequent Callers")
                st.plotly_chart(fig_top, use_container_width=True)

        # ---- Suspicious Caller Detection (Vectorized & Fast) ----
        elif selected_mode == "suspicious":
            st.subheader("🧩 Suspicious Caller Detection")
            if "caller" in df.columns and "timestamp" in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df['hour'] = df['timestamp'].dt.hour

                # Calls per hour per caller
                calls_per_hour = df.groupby(['caller', 'hour']).size().rename("calls_per_hour").reset_index()
                threshold_hour = calls_per_hour['calls_per_hour'].mean() + 2*calls_per_hour['calls_per_hour'].std()
                total_calls = df['caller'].value_counts()
                total_calls_threshold = total_calls.mean() + 2*total_calls.std()
                short_call_threshold = 5  # seconds

                # Vectorized flags
                high_density_callers = calls_per_hour[calls_per_hour['calls_per_hour'] > threshold_hour]['caller'].unique()
                odd_hour_callers = df[(df['hour'] < 6) | (df['hour'] > 22)]['caller'].unique()
                too_many_calls_callers = total_calls[total_calls > total_calls_threshold].index
                short_duration_callers = df.groupby('caller')['duration'].mean()
                short_duration_callers = short_duration_callers[short_duration_callers < short_call_threshold].index

                suspicious_callers_set = set(high_density_callers) | set(odd_hour_callers) | set(too_many_calls_callers) | set(short_duration_callers)

                if suspicious_callers_set:
                    st.markdown("### Phone Numbers Flagged as Suspicious:")
                    for caller in suspicious_callers_set:
                        reasons = []
                        illegal_risk = []

                        if caller in high_density_callers:
                            reasons.append("Excessive calls in a short period (possible spam/telemarketing).")
                            illegal_risk.append("Potential violation of telecom anti-spam regulations.")
                        if caller in odd_hour_callers:
                            reasons.append("Calls during odd hours (00:00–06:00 or late night).")
                            illegal_risk.append("Likely illegal cold-calling or harassment.")
                        if caller in too_many_calls_callers:
                            reasons.append("High total number of calls compared to others.")
                            illegal_risk.append("May indicate telemarketing abuse or automated calling.")
                        if caller in short_duration_callers:
                            reasons.append("Very short call durations (1–5 seconds).")
                            illegal_risk.append("Likely robocalls or scam attempts.")

                        st.markdown(f"**{caller}**")
                        for r in reasons:
                            st.markdown(f"- {r}")
                        st.markdown("**Potential illegal / spam activity:**")
                        for ir in illegal_risk:
                            st.markdown(f"- {ir}")
                        st.markdown("---")
                else:
                    st.info("No suspicious callers detected based on advanced metrics.")
            else:
                st.error("Required columns 'caller' and 'timestamp' not found.")

        # ---- Cross-Region Call Analysis ----
        elif selected_mode == "crossregion":
            st.subheader("🌐 Cross-Region Call Analysis")
            if "caller" in df.columns:
                geo_data = [{"Number": num, "City": city, "Country": country} 
                            for num in df["caller"].unique() 
                            for city, country in [get_location_from_number(num)]]
                geo_df = pd.DataFrame(geo_data)
                region_counts = geo_df.groupby('Country').size().reset_index(name='Count')
                fig_region = px.bar(region_counts, x='Country', y='Count', title="Calls per Country")
                st.plotly_chart(fig_region, use_container_width=True)
                st.write("Detailed Caller Regions:")
                st.dataframe(geo_df)

        # ---- AI Risk Analysis ----
        elif selected_mode == "ai":
            st.subheader("🧠 Gemini AI Insights")
            with st.spinner("Analyzing call patterns using Gemini..."):
                prompt = build_ai_prompt(uploaded.name, df)
                summary = call_gemini(prompt)
                st.markdown(summary)
                st.download_button("📄 Download AI Report", summary, file_name="CallInsightAI_Report.txt")

        # ---- Dataset Preview ----
        st.markdown("---")
        st.subheader("📄 Preview of Uploaded Call Data")
        st.dataframe(df.head(10))

    else:
        st.error("❌ Could not read uploaded file.")
else:
    st.info("⬆️ Upload a telecom call log file to begin analysis.")
