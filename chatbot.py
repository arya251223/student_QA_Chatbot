##############################################################
# AI-Powered Student Support Chatbot – Telegram Deployment
# ------------------------------------------------------------
# Features:
# 1) Intent matching with Sentence-BERT + cosine similarity
# 2) "general" fallback + unanswered.csv logging
# 3) /teach (admin): add new Q/A and rebuild embeddings live
# 4) /stats (admin): simple usage analytics
# 5) Spell-checker: corrects small typos in user input
##############################################################

# ---------- Imports ----------
import os  # Handles environment variables (for bot token, admin ID, file paths).
import json  # Reads/writes JSON files (used in stats.json for analytics).
from datetime import datetime  # Creates timestamps to log unanswered questions.
import pandas as pd  # Pandas library to load, clean, and save CSV datasets.
import torch  # Backend tensor operations (used internally by sentence-transformers).
from sentence_transformers import SentenceTransformer, util  
# SentenceTransformer = loads the pre-trained embedding model.
# util = utility functions, especially cosine similarity for comparing text embeddings.

from telegram import Update  # Object representing each Telegram message/update event.
from telegram.ext import (  
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
)  # Telegram bot framework: lets us define commands (/start, /help) and handle messages.

from dotenv import load_dotenv  # Loads TELEGRAM_BOT_TOKEN and ADMIN_USER_ID securely from .env file.
from spellchecker import SpellChecker  # Automatically detects and corrects small spelling mistakes.


# ---------- Config / Constants ----------
DATASET_PATH = "student_faq_chatbot_dataset.csv"   # Main dataset containing (intent, question, answer).
UNANSWERED_PATH = "unanswered.csv"                 # Stores questions bot couldn’t answer confidently.
STATS_PATH = "stats.json"                          # JSON file storing usage stats like answered/fallback.
MODEL_NAME = "all-MiniLM-L6-v2"                    # Small but accurate embedding model for intent matching.
SIM_THRESHOLD = 0.65                               # Similarity score threshold for deciding correct answer.


# ---------- Global State ----------
df = None                       # DataFrame holding the FAQ dataset after loading.
intents = []                    # List of intent labels from dataset.
questions = []                  # List of training questions (user queries).
answers = []                    # List of answers aligned with questions.
question_embeddings = None      # Embedding matrix of all dataset questions.
model = None                    # SentenceTransformer model instance.

spell = SpellChecker(distance=1)  # SpellChecker object, allows fixing words with 1 typo distance.
DOMAIN_WORDS = set()              # Domain-specific words that should not be autocorrected (e.g., AICTE).


##############################################################
# 1) Startup: environment, dataset, embeddings, spellchecker
##############################################################

def ensure_files_exist():
    """
    Ensures stats.json and unanswered.csv exist before starting the bot.
    - If stats.json does not exist, create a JSON file with counters (total, answered, fallback).
    - If unanswered.csv does not exist, create an empty CSV with headers for logs.
    This prevents FileNotFoundError when bot first runs.
    """
    if not os.path.exists(STATS_PATH):
        with open(STATS_PATH, "w", encoding="utf-8") as f:
            json.dump({"total": 0, "answered": 0, "fallback": 0, "per_intent": {}}, f, indent=2)

    if not os.path.exists(UNANSWERED_PATH):
        pd.DataFrame({"timestamp": [], "user_id": [], "username": [], "question": []}).to_csv(
            UNANSWERED_PATH, index=False
        )


def load_dataset(path: str):
    """
    Loads the FAQ dataset from CSV and performs cleaning.
    Steps:
    - Drop rows with missing intent/question/answer values.
    - Convert all text fields to strings and strip extra spaces.
    - Remove duplicate entries with the same intent/question/answer.
    Returns: A clean pandas DataFrame ready for training.
    """
    df_local = pd.read_csv(path)
    df_local = df_local.dropna(subset=["intent", "question", "answer"])  # Remove incomplete rows.
    df_local["intent"] = df_local["intent"].astype(str).str.strip()      # Normalize intent column.
    df_local["question"] = df_local["question"].astype(str).str.strip()  # Normalize questions.
    df_local["answer"] = df_local["answer"].astype(str).str.strip()      # Normalize answers.
    df_local = df_local.drop_duplicates(subset=["intent", "question", "answer"])  # Remove duplicates.
    return df_local


def rebuild_embeddings():
    """
    Creates or refreshes embeddings for all dataset questions.
    - Extract intents, questions, and answers from dataset (df).
    - Encode all questions into embeddings using the SentenceTransformer model.
    - These embeddings are stored in memory for fast cosine similarity checks.
    This is essential after every dataset update (like /teach).
    """
    global intents, questions, answers, question_embeddings
    intents = df["intent"].tolist()
    questions = df["question"].tolist()
    answers = df["answer"].tolist()

    question_embeddings = model.encode(
        questions, convert_to_tensor=True, normalize_embeddings=True
    )


def build_domain_words():
    """
    Collects all unique words from the dataset so spellchecker doesn’t wrongly correct them.
    - Example: 'AICTE' should not be corrected to 'ACTE'.
    - We gather all unique words from questions and add them to DOMAIN_WORDS.
    - These are added to SpellChecker’s dictionary to mark them as valid words.
    """
    global DOMAIN_WORDS
    words = set()
    for q in df["question"].tolist():
        for w in q.split():
            words.add(w.lower())  
    DOMAIN_WORDS = words | {"AICTE", "hostel", "canteen"}  
    for w in DOMAIN_WORDS:
        spell.word_frequency.add(w)  # Protect domain words from being corrected.


def correct_spelling(text: str) -> str:
    """
    Runs spell correction on a given text.
    - Splits user text into words.
    - If a word is domain-specific, numeric, or very short (<=2), don’t correct it.
    - Otherwise, correct it using SpellChecker.
    Returns: Corrected sentence (e.g., "libary timing" → "library timing").
    """
    corrected_words = []
    for word in text.split():
        lw = word.lower()
        if lw in DOMAIN_WORDS or lw.isdigit() or len(word) <= 2:
            corrected_words.append(word)  
        else:
            corrected = spell.correction(word)
            corrected_words.append(corrected if corrected else word)  # Fallback to original word
    return " ".join(corrected_words)


def init_all():
    """
    Initializes the entire chatbot system when program starts.
    Steps:
    1. Load .env variables (Telegram token, admin ID).
    2. Ensure stats.json and unanswered.csv exist.
    3. Load the embedding model into memory.
    4. Load and clean dataset.
    5. Build domain words for spellchecker.
    6. Build question embeddings for similarity matching.
    """
    load_dotenv()  
    ensure_files_exist()  
    global model, df
    model = SentenceTransformer(MODEL_NAME)  
    df = load_dataset(DATASET_PATH)  
    build_domain_words()  
    rebuild_embeddings()  


##############################################################
# 2) Core NLP: classify intent
##############################################################

def classify_intent(user_text: str):
    """
    Main logic for deciding which intent a user query matches.
    Steps:
    1. Correct spelling of the user query.
    2. Encode corrected text into an embedding vector.
    3. Compare it with all dataset question embeddings using cosine similarity.
    4. Pick the most similar question and check its score.
    5. If score >= SIM_THRESHOLD, return matching intent + answer.
    6. Otherwise, fallback to 'general' response and log unanswered.
    """
    user_text = correct_spelling(user_text)  
    user_emb = model.encode(user_text, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(user_emb, question_embeddings)[0]  
    best_idx = int(torch.argmax(sims))  
    best_score = float(sims[best_idx])  

    if best_score >= SIM_THRESHOLD:
        return intents[best_idx], answers[best_idx], best_score, best_idx
    else:
        fallback = (
            "I’m not fully sure about that. I’ll note this and improve later. "
            "Please rephrase your question."
        )
        return "general", fallback, best_score, None

##############################################################
# 3) Logging + Analytics
##############################################################

def log_unanswered(user_id: int, username: str, question: str):
    """
    Logs any user question that the bot cannot answer confidently.
    Why?
    - This gives admins visibility into what students are asking that is missing from dataset.
    - We save timestamp, user ID, username, and the actual question.
    - Stored in unanswered.csv so it can later be used to improve training data.
    """
    ts = datetime.utcnow().isoformat()  # Current time in UTC (ISO format for consistency).
    row = {"timestamp": ts, "user_id": user_id, "username": username or "", "question": question}
    df_log = pd.DataFrame([row])  # Wrap row into a pandas DataFrame for easy saving.
    
    # Check if unanswered.csv exists and has content.
    # If file is empty/new → include headers. If not → append without headers.
    header_needed = not os.path.exists(UNANSWERED_PATH) or os.path.getsize(UNANSWERED_PATH) == 0
    
    # Append the row into unanswered.csv
    df_log.to_csv(UNANSWERED_PATH, mode="a", index=False, header=header_needed)


def bump_stats(intent: str):
    """
    Updates usage statistics stored in stats.json.
    Purpose:
    - Track how many questions were asked.
    - Track how many were answered vs fallback (general).
    - Track per-intent frequency (e.g., 'fees': 50 times).
    This helps admins understand usage patterns and weak spots.
    """
    # Load existing stats.json (dictionary format).
    with open(STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)

    stats["total"] = stats.get("total", 0) + 1  # Count every question asked.

    if intent == "general":  # If bot gave fallback.
        stats["fallback"] = stats.get("fallback", 0) + 1
    else:  # If bot matched a real intent.
        stats["answered"] = stats.get("answered", 0) + 1
        per = stats.get("per_intent", {})
        per[intent] = per.get(intent, 0) + 1  # Increment counter for that intent.
        stats["per_intent"] = per

    # Save updated stats.json
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

##############################################################
# 4) Telegram Handlers (async for python-telegram-bot v20+)
##############################################################

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Responds to /start command.
    Purpose:
    - Give a warm introduction.
    - Explain what bot can do.
    - Highlight admin-only commands.
    """
    name = update.effective_user.first_name or "there"  # Grab user's first name for personalization.
    intro = (
        f"👋 Hi {name}! I’m your *Student Support Chatbot*.\n\n"
        "I can help with admissions, fees, exams, hostel, library, and more.\n"
        "• Ask me anything (e.g., *What are library timings?*)\n"
        "• If I’m not sure, I’ll note the question for improvement.\n\n"
        "_Admin tools_: /teach  /stats"
    )
    # Reply with MarkdownV2 formatting (bold, italics, emoji).
    await update.message.reply_text(intro)



async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Responds to /help command.
    Purpose:
    - Show features clearly.
    - Provide usage examples.
    - Show admin-only commands separately.
    """
    text = (
        "Here’s what I can do:\n"
        "• Answer common student queries (admissions, fees, exams, hostel, library…)\n"
        "• If I’m unsure, I’ll save the question for the team to improve me.\n\n"
        "Examples:\n"
        "• *When are exam forms due?*\n"
        "• *How to apply for a hostel?*\n\n"
        "_Admin commands_\n"
        "• /teach <question> || <intent> || <answer>\n"
        "• /stats"
    )
    await update.message.reply_text(text)


def is_admin(update: Update) -> bool:
    """
    Helper function that checks if user is admin.
    - Reads ADMIN_USER_ID from .env file.
    - Compares with current user's Telegram ID.
    - Only admin can run sensitive commands (/teach, /stats).
    """
    admin_id = os.getenv("ADMIN_USER_ID", "").strip()
    return admin_id and str(update.effective_user.id) == admin_id


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Responds to /stats command (admin only).
    Purpose:
    - Shows overall usage analytics.
    - Total, answered, fallback.
    - Per-intent breakdown.
    """
    if not is_admin(update):
        return await update.message.reply_text("Admin only.")

    with open(STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)

    total = stats.get("total", 0)
    answered = stats.get("answered", 0)
    fallback = stats.get("fallback", 0)
    per_intent = stats.get("per_intent", {})

    lines = [
        f"📊 *Stats*",
        f"Total: {total}",
        f"Answered: {answered}",
        f"Fallback (general): {fallback}",
        "—— Per Intent ——",
    ] + [f"{k}: {v}" for k, v in sorted(per_intent.items(), key=lambda x: -x[1])]

    await update.message.reply_markdown("\n".join(lines))

import shlex
async def cmd_teach(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Responds to /teach command (admin only).
    Usage:
    /teach Question || intent || Answer
    Purpose:
    - Add new Q/A to dataset.
    - Rebuild embeddings so bot learns instantly.
    """
    if not is_admin(update):
        return await update.message.reply_text("Admin only.")

    if not context.args:
        return await update.message.reply_text(
            'Usage:\n/teach "Question" "Intent" "Answer"\n\n'
            'Example:\n/teach "admission process?" "campus_info" "Just visit administrative office or Admission Dept and they will guide you"'
        )

    # Join args back into a single string (since Telegram splits by spaces)
    payload = " ".join(context.args)

    try:
        # Parse quoted arguments
        parts = shlex.split(payload)
    except ValueError:
        return await update.message.reply_text("❌ Could not parse input. Make sure to use quotes around each part.")

    if len(parts) != 3:
        return await update.message.reply_text("❌ Please provide exactly 3 parts: \"Question\" \"Intent\" \"Answer\"")

    new_q, new_intent, new_ans = parts

    # Append row into dataset CSV
    new_row = pd.DataFrame([{"question": new_q, "intent": new_intent, "answer": new_ans}])
    new_row.to_csv(DATASET_PATH, mode="a", index=False, header=not os.path.exists(DATASET_PATH))

    # Reload dataset and rebuild embeddings
    global df
    df = load_dataset(DATASET_PATH)
    rebuild_embeddings()

    await update.message.reply_text("✅ Learned! New example added and model refreshed.")

##############################################################
# 5) User Messages (normal text, not commands)
##############################################################

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles regular user messages.
    Flow:
    1. Read user message text.
    2. Pass it through classify_intent.
    3. If classified as 'general', log unanswered.
    4. Update analytics counters.
    5. Reply back with answer or fallback.
    """
    user_text = (update.message.text or "").strip()
    if not user_text:  # Ignore empty messages.
        return

    intent, answer, score, best_idx = classify_intent(user_text)

    # If no confident match, log it.
    if intent == "general":
        log_unanswered(update.effective_user.id, update.effective_user.username, user_text)

    bump_stats(intent)  # Update stats.json counters.

    reply = answer
    # For debugging, you could add confidence score:
    # reply += f"\n\n(_confidence: {score:.2f}_)"
    await update.message.reply_text(reply)
##############################################################
# 6) Main: run Telegram bot
##############################################################

def main():
    """
    Program entry point.
    Steps:
    1. Initialize everything (load dataset, model, embeddings).
    2. Load Telegram bot token from .env.
    3. Create Application (async handler).
    4. Register command + message handlers.
    5. Start polling Telegram for messages.
    """
    init_all()

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN. Put it in .env.")

    app = ApplicationBuilder().token(token).build()

    # Register commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("teach", cmd_teach))

    # Register normal messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    print("✅ Bot is live. Press Ctrl+C to stop.")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()


