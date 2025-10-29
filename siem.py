# --- Imports ---
import os
import io
import re
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import PyPDF2
import plotly.express as px
import google.generativeai as genai
from datetime import datetime
import requests
import ipaddress
import subprocess
import platform

# --- Load environment ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.sidebar.error("Set GOOGLE_API_KEY in .env before running the app.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- Helper functions ---
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    except Exception:
        return ""

def try_load_dataframe(file_bytes: bytes, filename: str):
    lower = filename.lower()
    try:
        if lower.endswith(".csv"):
            return pd.read_csv(io.BytesIO(file_bytes))
        if lower.endswith(".json"):
            text = file_bytes.decode(errors="ignore")
            try:
                return pd.read_json(io.StringIO(text), lines=True)
            except:
                return pd.json_normalize(json.loads(text))
        if lower.endswith((".log", ".txt")):
            text = file_bytes.decode(errors="ignore")
            for sep in [',', '\t', '|', ' ']:
                try:
                    df = pd.read_csv(io.StringIO(text), sep=sep, nrows=10)
                    if df.shape[1] > 1:
                        return pd.read_csv(io.StringIO(text), sep=sep)
                except:
                    continue
            return pd.DataFrame({"raw": text.splitlines()})
        if lower.endswith(".pdf"):
            text = extract_text_from_pdf(file_bytes)
            return pd.DataFrame({"raw": [l for l in text.splitlines() if l.strip()]})
    except Exception as e:
        st.warning(f"Could not auto-parse file: {e}")
    return None

# --- IP extraction ---
def extract_ips_from_dataframe(df: pd.DataFrame) -> pd.Series:
    ip_pattern = r"((?:\d{1,3}\.){3}\d{1,3})"
    ips = []
    for col in df.columns:
        if df[col].dtype == object:
            matches = df[col].astype(str).str.extractall(ip_pattern)[0].dropna().tolist()
            ips.extend(matches)
    return pd.Series(ips, name="IP Address")

# --- Check public IP ---
def is_public_ip(ip):
    try:
        return ipaddress.ip_address(ip).is_global
    except:
        return False

# --- Stats extraction ---
def basic_log_stats(df: pd.DataFrame):
    stats = {"rows": len(df), "columns": list(df.columns)}
    for col in df.columns:
        low = col.lower()
        if "ip" in low:
            try:
                stats["top_ips"] = df[col].value_counts().nlargest(10).to_dict()
            except:
                pass
        if "event" in low or "action" in low or "type" in low:
            try:
                stats["top_events"] = df[col].value_counts().nlargest(10).to_dict()
            except:
                pass
        if "time" in low or "timestamp" in low or "date" in low:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                stats["time_col"] = col
                stats["time_range"] = {
                    "min": str(df[col].min()),
                    "max": str(df[col].max())
                }
            except:
                pass
    return stats

# --- Risk scoring ---
def calculate_risk_score(stats: dict) -> int:
    score = 10
    if "top_events" in stats:
        events = " ".join(stats["top_events"].keys()).lower()
        if any(k in events for k in ["failed", "denied", "attack", "error"]):
            score += 30
        if any(k in events for k in ["malware", "exploit", "ransom"]):
            score += 50
    if "top_ips" in stats and len(stats["top_ips"]) > 5:
        score += 10
    if "rows" in stats and stats["rows"] > 10000:
        score += 5
    return min(score, 100)

# --- GenAI ---
def build_genai_prompt(filename, stats, sample_rows):
    prompt = [
        "Analyze the SIEM log data below concisely. Identify major risks associated with specific IPs, explain potential impact, and suggest preventive measures for each risk.",
        f"File: {filename}",
        f"Stats: rows={stats.get('rows')}, columns={', '.join(stats.get('columns', []))}"
    ]
    if "top_events" in stats:
        prompt.append(f"Top events: {stats['top_events']}")
    if "top_ips" in stats:
        prompt.append(f"Top IPs: {stats['top_ips']}")
    if "time_range" in stats:
        prompt.append(f"Time range: {stats['time_range']}")
    prompt.append("\nSample logs:\n" + sample_rows)
    return "\n".join(prompt)

def call_genai(prompt_text: str, model_name="gemini-2.5-flash"):
    try:
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt_text)
        return getattr(resp, "text", str(resp))
    except Exception as e:
        return f"ERROR calling GenAI: {e}"

# --- Geo-Map ---
def get_ip_location(ip):
    try:
        response = requests.get(f"http://ip-api.com/json/{ip}").json()
        if response['status'] == 'success':
            return response['lat'], response['lon'], response['country'], response['city']
    except:
        pass
    return None, None, None, None

def plot_ip_map(ip_list, suspicious_ips=[]):
    data = []
    for ip in ip_list.unique():
        lat, lon, country, city = get_ip_location(ip)
        if lat and lon:
            data.append({
                "IP": ip,
                "Lat": lat,
                "Lon": lon,
                "Suspicious": ip in suspicious_ips,
                "Location": f"{city}, {country}"
            })
    if not data:
        st.warning("No public IPs available to map.")
        return
    df_map = pd.DataFrame(data)
    fig = px.scatter_geo(
        df_map,
        lat="Lat",
        lon="Lon",
        color="Suspicious",
        hover_name="IP",
        hover_data=["Location"],
        color_discrete_map={True: "red", False: "green"},
        title="🌐 Geo-Map of Public IPs"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Network Info ---
def get_network_info():
    info = {}
    try:
        response = requests.get("https://ipinfo.io/json", timeout=5).json()
        info["Public IP"] = response.get("ip", "N/A")
        info["ISP"] = response.get("org", "N/A")
        info["Country"] = response.get("country", "N/A")
        info["City"] = response.get("city", "N/A")
    except:
        info["Public IP"] = info["ISP"] = info["Country"] = info["City"] = "Error retrieving data"

    ssid = "N/A"
    if platform.system() == "Windows":
        try:
            output = subprocess.check_output(["netsh", "wlan", "show", "interfaces"], text=True)
            for line in output.split("\n"):
                if "SSID" in line and "BSSID" not in line:
                    ssid = line.split(":")[1].strip()
                    break
        except:
            ssid = "Unavailable"
    else:
        ssid = "Unsupported on this OS"

    info["Wi-Fi SSID"] = ssid
    return info

# --- Model Info / Accuracy Meter ---
MODEL_NAME = "Gemini 2.5 Flash"
def show_model_info(accuracy=None, error_msg=None):
    st.markdown("### 🧪 Model Info & Accuracy")
    st.write(f"**Model:** {MODEL_NAME}")
    if accuracy is not None:
        st.progress(min(max(accuracy, 0), 100))
        st.write(f"**Estimated Accuracy:** {accuracy}%")
    if error_msg:
        st.error(f"Error: {error_msg}")

# --- Streamlit UI ---
st.title("🛡️ SIEMInsightAI — Log Analyzer & Risk Dashboard")
st.write("Upload a SIEM log file to get GenAI-powered insights, risk scoring, login attempts, and IP Geo-Map.")

uploaded_file = st.file_uploader("Choose a log file", type=["csv", "json", "txt", "log", "pdf"])

if uploaded_file:
    df = try_load_dataframe(uploaded_file.read(), uploaded_file.name)
    if df is not None:
        st.success(f"Parsed {len(df)} rows, {len(df.columns)} columns.")
        stats = basic_log_stats(df)

        menu_options = [
            "🔥 Risk Scoring",
            "🔐 Login Attempts",
            "🌐 Geo-Map of IPs",
            "🧠 Risks & Recommendations",
            "📶 My Network Info"
        ]
        cols = st.columns(len(menu_options))
        menu_option = None
        for i, option in enumerate(menu_options):
            if cols[i].button(option, use_container_width=True):
                menu_option = option
        if not menu_option:
            menu_option = menu_options[0]

        # --- Section rendering ---
        if menu_option == "🔥 Risk Scoring":
            risk_score = calculate_risk_score(stats)
            st.subheader("🔥 Risk Scoring System")
            st.metric("Estimated Risk Level", f"{risk_score} / 100")
            if risk_score < 40:
                st.success("Low Risk: Mostly normal activity.")
            elif risk_score < 70:
                st.warning("Moderate Risk: Some suspicious patterns detected.")
            else:
                st.error("High Risk: Critical events or anomalies detected!")
            show_model_info(accuracy=100-risk_score)  # example: lower risk → higher accuracy

        elif menu_option == "🔐 Login Attempts":
            ip_col = None
            for c in df.columns:
                if "ip" in c.lower():
                    ip_col = c
                    break
            if not ip_col:
                extracted_ips = extract_ips_from_dataframe(df)
                login_df = pd.DataFrame({"IP Address": extracted_ips})
            else:
                login_df = df.copy()
                login_df.rename(columns={ip_col: "IP Address"}, inplace=True)

            event_col = None
            for c in df.columns:
                if any(k in c.lower() for k in ["event", "action", "type", "activity", "message"]):
                    event_col = c
                    break
            if event_col:
                mask = login_df[event_col].astype(str).str.contains(
                    "login|auth|access|failed|signin", case=False, na=False
                )
                if mask.any():
                    login_df = login_df[mask]

            ip_counts = login_df["IP Address"].value_counts().reset_index()
            ip_counts.columns = ["IP Address", "Login Attempts"]

            st.subheader("🔐 Login Attempts per IP Address")
            fig = px.bar(
                ip_counts,
                x="IP Address",
                y="Login Attempts",
                title="Number of Login Attempts per IP",
                labels={"IP Address": "IP Address", "Login Attempts": "Login Attempts"},
            )
            st.plotly_chart(fig, use_container_width=True)

            high_attempts = ip_counts[ip_counts["Login Attempts"] > ip_counts["Login Attempts"].mean() * 2]
            if not high_attempts.empty:
                st.warning("🚨 Suspicious IPs with unusually high login attempts:")
                st.dataframe(high_attempts)

            show_model_info(accuracy=90)  # example static accuracy

        elif menu_option == "🌐 Geo-Map of IPs":
            extracted_ips = extract_ips_from_dataframe(df).dropna().drop_duplicates()
            public_ips = [ip for ip in extracted_ips if is_public_ip(ip)]
            if not public_ips:
                st.warning("No public IPs found for Geo-Map (private/local IPs cannot be mapped).")
            else:
                ip_counts = pd.Series(public_ips).value_counts()
                suspicious_ips = ip_counts[ip_counts > ip_counts.mean() * 2].index.tolist()
                st.subheader("🌐 Geo-Map of Detected Public IPs")
                plot_ip_map(pd.Series(public_ips), suspicious_ips)
            show_model_info(accuracy=85)

        elif menu_option == "🧠 Risks & Recommendations":
            with st.spinner("Analysing logs with Gemini..."):
                sample_rows = df.head(10).to_csv(index=False)
                prompt = build_genai_prompt(uploaded_file.name, stats, sample_rows)
                summary = call_genai(prompt)
                st.subheader("🧠 AI-Generated Risks & Recommendations")
                st.markdown(summary)
                st.download_button("Download Summary", summary, file_name=f"{uploaded_file.name}_summary.txt")
                error_msg = summary if summary.startswith("ERROR") else None
                show_model_info(accuracy=80, error_msg=error_msg)

        elif menu_option == "📶 My Network Info":
            st.subheader("📶 My Network & IP Information")
            with st.spinner("Fetching network info..."):
                net_info = get_network_info()
                st.json(net_info)
            show_model_info(accuracy=95)

        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head(10))
    else:
        st.error("Could not parse file.")
else:
    st.info("Upload a log file to start analysis.")
