import os
import re
import pandas as pd
from flask import Flask, request, jsonify
from groq import Groq
from dotenv import load_dotenv
import tiktoken
from datetime import datetime

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = Flask(__name__)

CSV_FILE = "chat_summary_log.csv"
REQUIRED_COLUMNS = [
    "Conversation", "Summary", "Behavior Eval", "Behavior Score",
    "Conversation Eval", "Conversation Score", "Know-how Eval",
    "Know-how Score", "Agent Reported", "Timestamp UTC",
    "Date (IST)", "Time (IST)"
]

# Ensure CSV has headers
if not os.path.exists(CSV_FILE) or os.stat(CSV_FILE).st_size == 0:
    pd.DataFrame(columns=REQUIRED_COLUMNS).to_csv(CSV_FILE, index=False)

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

def estimate_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def detect_sensitive_info(convo):
    red_flags = ["credit card", "password", "cvv", "otp", "pin", "ssn", "account number"]
    convo_lower = convo.lower()
    return any(flag in convo_lower for flag in red_flags)

def evaluate_convo(convo):
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": convo}
        ]
    )
    return response.choices[0].message.content

def extract_info(groq_output):
    summary = behavior = conv = know = ""
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
                behavior = match.group(1).strip()
                sb = int(match.group(2))
        elif "- conversation quality:" in line_lower:
            current = "conv"
            match = re.search(r"- Conversation Quality:\s*(.*)\(Score:\s*(\d)/5\)", line, re.IGNORECASE)
            if match:
                conv = match.group(1).strip()
                sc = int(match.group(2))
        elif "- know-how of the issue:" in line_lower:
            current = "know"
            match = re.search(r"- Know-How of the Issue:\s*(.*)\(Score:\s*(\d)/5\)", line, re.IGNORECASE)
            if match:
                know = match.group(1).strip()
                sk = int(match.group(2))
        else:
            if current == "summary":
                summary += line + " "

    return summary.strip(), behavior, conv, know, sb, sc, sk

def log_to_csv(convo, summary, behavior, conv, know, sb, sc, sk, flagged):
    timestamp_utc = pd.to_datetime(datetime.utcnow(), utc=True)
    timestamp_ist = timestamp_utc.tz_convert("Asia/Kolkata")
    row = {
        "Conversation": convo,
        "Summary": summary,
        "Behavior Eval": behavior,
        "Behavior Score": sb,
        "Conversation Eval": conv,
        "Conversation Score": sc,
        "Know-how Eval": know,
        "Know-how Score": sk,
        "Agent Reported": flagged,
        "Timestamp UTC": timestamp_utc.isoformat(),
        "Date (IST)": timestamp_ist.date(),
        "Time (IST)": timestamp_ist.time()
    }
    df = pd.DataFrame([row])
    df.to_csv(CSV_FILE, mode="a", header=False, index=False)

@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.json
    convo = data.get("conversation", "")
    if not convo.strip():
        return jsonify({"error": "Conversation is required"}), 400

    try:
        tokens_estimated = estimate_tokens(convo)
        flagged = detect_sensitive_info(convo)
        timestamp_utc = datetime.utcnow().isoformat()

        output = evaluate_convo(convo)
        summary, behavior, conv, know, sb, sc, sk = extract_info(output)

        # Save to CSV
        log_to_csv(convo, summary, behavior, conv, know, sb, sc, sk, flagged)

        return jsonify({
            "tokens_estimated": tokens_estimated,
            "agent_reported": flagged,
            "timestamp_utc": timestamp_utc,
            "summary": summary,
            "evaluation": {
                "behavior": {"text": behavior, "score": sb},
                "conversation_quality": {"text": conv, "score": sc},
                "know_how": {"text": know, "score": sk}
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
