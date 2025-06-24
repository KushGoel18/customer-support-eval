import os
import re
import streamlit as st
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CSV_FILE = "chat_summary_log.csv"

# Ensure CSV has correct headers if cleared or missing
required_columns = [
    "Conversation", "Summary", "Behavior Eval", "Behavior Score",
    "Conversation Eval", "Conversation Score", "Know-how Eval",
    "Know-how Score", "Agent Reported", "Timestamp UTC",
    "Date (IST)", "Time (IST)"
]
if not os.path.exists(CSV_FILE) or os.stat(CSV_FILE).st_size == 0:
    pd.DataFrame(columns=required_columns).to_csv(CSV_FILE, index=False)

# following is my prompt template 
system_prompt = """
You are an expert evaluator of customer support chats. Based on the conversation below:
1. Summarize the conversation.
2. Evaluate the agent's behavior: professionalism, tone, empathy.
3. Evaluate the agent's conversation handling: clarity, responsiveness, and structure.
4. Evaluate the agent's knowledge of the issue: correctness, understanding, and resolution offered.
5. For each evaluation category, also give a score from 1 to 5 (5 being best).

Also, if the agent ever asks for sensitive information such as credit card number, password, CVV, SSN, or OTP, consider it a serious violation and give a Behavior score of 1, and mention that it is against company policy.

Give your response in the following format:

Summary:
[Your summary]

Agent Evaluation:
- Behavior: [Textual evaluation] (Score: X/5)
- Conversation Quality: [Textual evaluation] (Score: X/5)
- Know-How of the Issue: [Textual evaluation] (Score: X/5)
"""

def evaluate_conversation(convo):
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": convo}
        ]
    )
    return response.choices[0].message.content

def extract_info(groq_output):
    summary = behavior_eval = conv_eval = knowhow_eval = ""
    sb = sc = sk = 1
    lines = groq_output.splitlines()
    current = None

    for line in lines:
        line_lower = line.strip().lower()
        if line_lower.startswith("summary:"):
            current = "summary"
            summary = ""
        elif "- behavior:" in line_lower:
            current = "behavior"
            match = re.search(r"- Behavior:\s*(.*)\(Score:\s*(\d)/5\)", line, re.IGNORECASE)
            if match:
                behavior_eval = match.group(1).strip()
                sb = int(match.group(2))
        elif "- conversation quality:" in line_lower:
            current = "conv"
            match = re.search(r"- Conversation Quality:\s*(.*)\(Score:\s*(\d)/5\)", line, re.IGNORECASE)
            if match:
                conv_eval = match.group(1).strip()
                sc = int(match.group(2))
        elif "- know-how of the issue:" in line_lower:
            current = "know"
            match = re.search(r"- Know-How of the Issue:\s*(.*)\(Score:\s*(\d)/5\)", line, re.IGNORECASE)
            if match:
                knowhow_eval = match.group(1).strip()
                sk = int(match.group(2))
        else:
            if current == "summary":
                summary += line + " "

    return summary.strip(), behavior_eval, conv_eval, knowhow_eval, sb, sc, sk

def detect_sensitive_info(convo):
    red_flags = ["credit card", "password", "cvv", "otp", "pin", "ssn", "account number"]
    convo_lower = convo.lower()
    return any(flag in convo_lower for flag in red_flags)

def cache_result(convo, summary, beh, conv, know, sb, sc, sk):
    agent_flagged = detect_sensitive_info(convo)
    timestamp_utc = pd.to_datetime(datetime.utcnow(), utc=True)
    timestamp_ist = timestamp_utc.tz_convert("Asia/Kolkata")
    data = {
        "Conversation": convo,
        "Summary": summary,
        "Behavior Eval": beh,
        "Behavior Score": sb,
        "Conversation Eval": conv,
        "Conversation Score": sc,
        "Know-how Eval": know,
        "Know-how Score": sk,
        "Agent Reported": agent_flagged,
        "Timestamp UTC": timestamp_utc.isoformat(),
        "Date (IST)": timestamp_ist.date(),
        "Time (IST)": timestamp_ist.time()
    }
    df = pd.DataFrame([data])
    df.to_csv(CSV_FILE, mode="a", header=False, index=False)

# ------------------ UI ------------------
st.set_page_config(page_title="Support Chat Evaluator", layout="wide")
st.title("📟 Customer Support Ticket Summarizer")

with st.form("chat_form"):
    uploaded_file = st.file_uploader("📂 Or upload a .txt file with the conversation", type=["txt"])
    convo_input = ""

    if uploaded_file is not None:
        convo_input = uploaded_file.read().decode("utf-8")
        st.success("Conversation loaded from file!")
    else:
        convo_input = st.text_area("📤 Paste the customer-agent conversation:", height=300)

    submitted = st.form_submit_button("Analyze Conversation")

if submitted and convo_input.strip():
    with st.spinner("Evaluating via Groq..."):
        output = evaluate_conversation(convo_input)
        summary, beh_eval, conv_eval, know_eval, sb, sc, sk = extract_info(output)

        st.success("✅ Evaluation complete!")

        st.subheader("📌 Summary")
        st.write(summary)

        st.subheader("📊 Agent Evaluation")
        st.write(f"**Behavior**: {beh_eval} _(Score: {sb}/5)_")
        st.write(f"**Conversation Quality**: {conv_eval} _(Score: {sc}/5)_")
        st.write(f"**Know-How**: {know_eval} _(Score: {sk}/5)_")

        cache_result(convo_input, summary, beh_eval, conv_eval, know_eval, sb, sc, sk)

        with st.expander("🗂️ View all cached evaluations"):
            if os.path.exists(CSV_FILE):
                try:
                    df = pd.read_csv(CSV_FILE)
                    st.dataframe(df)
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")

        with st.expander("🚨 View flagged cases for audit"):
            if os.path.exists(CSV_FILE):
                try:
                    df = pd.read_csv(CSV_FILE)
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
                    st.stop()

                if "Agent Reported" not in df.columns:
                    df["Agent Reported"] = False
                if "Timestamp UTC" not in df.columns:
                    df["Timestamp UTC"] = pd.NaT

                df["Timestamp UTC"] = pd.to_datetime(df["Timestamp UTC"], errors="coerce", utc=True)
                df["Timestamp IST"] = df["Timestamp UTC"].dt.tz_convert("Asia/Kolkata")
                df["Date (IST)"] = df["Timestamp IST"].dt.date
                df["Time (IST)"] = df["Timestamp IST"].dt.time

                df_flagged = df[df["Agent Reported"] == True].dropna(subset=["Timestamp IST"])

                if not df_flagged.empty:
                    st.warning("Flagged agent behavior detected in the following cases:")
                    st.markdown("#### 📅 Filter by Date and Time Range (IST)")

                    min_date = df_flagged["Date (IST)"].min()
                    max_date = df_flagged["Date (IST)"].max()
                    min_time = df_flagged["Time (IST)"].min()
                    max_time = df_flagged["Time (IST)"].max()

                    start_date = st.date_input("Start Date", value=min_date)
                    start_time = st.time_input("Start Time", value=min_time)
                    end_date = st.date_input("End Date", value=max_date)
                    end_time = st.time_input("End Time", value=max_time)

                    start_dt = pd.Timestamp.combine(start_date, start_time).tz_localize("Asia/Kolkata")
                    end_dt = pd.Timestamp.combine(end_date, end_time).tz_localize("Asia/Kolkata")

                    filtered = df_flagged[
                        (df_flagged["Timestamp IST"] >= start_dt) &
                        (df_flagged["Timestamp IST"] <= end_dt)
                    ]

                    st.dataframe(filtered)
                    st.download_button("📥 Download Filtered Report (CSV)", data=filtered.to_csv(index=False),
                                       file_name="flagged_audit.csv", mime="text/csv")
                else:
                    st.success("No flagged agent behavior found.")
