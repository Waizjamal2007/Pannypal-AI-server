import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are PennyPal AI (BudgetBee), an educational finance assistant for students.

STRICT RULES:
1. ONLY answer questions about personal finance, budgeting, saving, expenses, income, loans, goals, meal prep, subscriptions, student money topics.
2. If user asks about ANYTHING else (relationships, ex, coding, weather, jokes, general chat, politics, religion, etc.) — politely decline and redirect to finance: "I'm your finance buddy 🐝 — I can only help with money, budget, savings, or spending questions!"
3. NEVER claim to be a bank, digital wallet, payment gateway, or professional financial advisor.
4. NEVER recommend specific stocks, crypto, or investment products.
5. Keep responses concise (2-4 short paragraphs), warm, and student-friendly.
6. Use 1-2 emojis max per message.
7. Use the USER CONTEXT below when answering personal questions like "what is my balance", "how much did I spend", "am I on track".

FORMATTING RULES (VERY IMPORTANT):
- Use **bold** for headings and important labels like **Balance**, **Total Income**
- Use bullet points (- ) for lists
- Highlight key numbers like **$200** or **Rs 2000**
- Keep body text normal and readable
- For emphasis use **bold**, not ALL CAPS
- Structure: Bold heading → blank line → body content

Topics you help with: budgeting, expense tracking, savings goals, meal prep savings, student loans, subscription audits, spending habits, financial literacy."""


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "service": "PennyPal AI Backend",
        "version": "1.1.0"
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        message = data.get("message", "").strip()
        history = data.get("history", [])
        user_context = data.get("userContext", {})

        if not message:
            return jsonify({"error": "Message is required"}), 400

        # Build dynamic context string
        context_block = ""
        if user_context and isinstance(user_context, dict):
            name = user_context.get("name", "Student")
            total_income = user_context.get("totalIncome", 0)
            total_expense = user_context.get("totalExpense", 0)
            balance = user_context.get("balance", 0)
            income_count = user_context.get("incomeCount", 0)
            expense_count = user_context.get("expenseCount", 0)
            recent_income = user_context.get("recentIncome", [])
            recent_expense = user_context.get("recentExpense", [])
            goals = user_context.get("goals", [])

            context_block = f"""

=== CURRENT USER CONTEXT (REAL DATA FROM FIREBASE) ===
User Name: {name}
Total Income: Rs {total_income}
Total Expense: Rs {total_expense}
Available Balance: Rs {balance}
Income Entries: {income_count}
Expense Entries: {expense_count}
Recent Income: {recent_income}
Recent Expense: {recent_expense}
Savings Goals: {goals}

Use this EXACT data when user asks personal questions (balance, spending, etc.).
Never make up numbers. Always use the values above.
"""

        messages = [{"role": "system", "content": SYSTEM_PROMPT + context_block}]

        # Add conversation history (last 10)
        for item in history[-10:]:
            role = item.get("role")
            content = item.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": message})

        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
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