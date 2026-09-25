import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # Flutter app se requests allow karne ke liye

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# System prompt - PennyPal personality
SYSTEM_PROMPT = """You are PennyPal AI (also called BudgetBee), a friendly and helpful personal finance assistant for students.

Your personality:
- Warm, encouraging, and slightly playful (you love bee puns occasionally)
- Practical and specific with numbers and advice
- Student-focused: understand tight budgets, meal prep, campus life
- Keep responses concise (2-4 short paragraphs max unless asked for detail)
- Use emojis sparingly (1-2 per message)
- When giving money advice, be specific with amounts and timelines
- Always encourage good habits

Topics you help with:
- Budgeting and expense tracking
- Saving goals (like buying a MacBook, trips, etc.)
- Meal prep and grocery savings
- Student loan planning
- Subscription audits
- Coffee/food spending habits

Never give investment advice or recommend specific stocks/crypto. Keep it simple and actionable."""


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "service": "PennyPal AI Backend",
        "version": "1.0.0"
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        message = data.get("message", "").strip()
        history = data.get("history", [])  # list of {role, content}

        if not message:
            return jsonify({"error": "Message is required"}), 400

        # Build messages array for Groq
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add previous conversation history (optional, last 10 messages)
        for item in history[-10:]:
            role = item.get("role")
            content = item.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        # Add current user message
        messages.append({"role": "user", "content": message})

        # Call Groq
        completion = client.chat.completions.create(
           model="openai/gpt-oss-120b",  # fast + smart
            messages=messages,
            temperature=0.7,
            max_tokens=800,
            top_p=0.9,
        )

        reply = completion.choices[0].message.content

        return jsonify({
            "success": True,
            "reply": reply,
            "model": completion.model,
        })

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)