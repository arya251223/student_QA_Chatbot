# Student Support Chatbot – Telegram Deployment

## 📌 Project Overview
This project is an **AI-powered Student Support Chatbot** designed to assist students with common queries related to **admissions, exams, hostel, library, and other academic services**.  
The chatbot leverages **Sentence-BERT** for semantic similarity and is deployed on **Telegram** for real-time interaction.

---

## ✅ Features
- Intent Classification using **Sentence-BERT (all-MiniLM-L6-v2)**
- **Cosine Similarity** for query–answer matching
- **Fallback** response for unknown queries
- Auto-logging of unanswered queries → `unanswered.csv`
- Self-learning via `/teach` (Admin only)
- Basic analytics via `/stats`
- **Spelling correction** for improved accuracy
- Fully deployed as a **Telegram Bot**

---

## 📂 Project Structure
```
├── chatbot.py                     # Main chatbot code
├── student_faq_chatbot_dataset.csv # FAQ dataset (question, intent, answer)
├── unanswered.csv                  # Logs unanswered questions
├── stats.json                      # Stores usage analytics
├── requirements.txt                # Required dependencies
├── .env                            # Environment variables (Bot Token, Admin ID)
├── LICENSE                         # Custom private license
└── README.md                       # Documentation
```

---

## ⚙️ Tech Stack
- **Python 3.10+**
- **Hugging Face Sentence-Transformers**
- **Telegram Bot API**
- **pandas, torch**
- **dotenv** for environment variables
- **pyspellchecker** for typo correction

---

## 🔍 How It Works
1. User sends a query via Telegram.  
2. Query is embedded using **Sentence-BERT**.  
3. The bot computes **cosine similarity** with stored FAQ embeddings.  
4. If similarity > threshold → returns the best answer.  
5. Otherwise → returns fallback and logs the query in `unanswered.csv`.  
6. Admin can **teach new Q&A** dynamically via `/teach`.  
7. Usage **analytics** available via `/stats`.  

---

## 📊 Dataset Format
The chatbot expects a CSV dataset in the following format:

```csv
question,intent,answer
What are the library timings?,library_timing,The library is open from 9 AM to 6 PM on weekdays.
How to apply for a hostel?,hostel_apply,You can apply via the Hostel Portal at ...
When are the semester fees due?,fees_due,Fees are due by ...
```

⚠️ **Order must be**: `question → intent → answer`.

---

## ▶️ How to Run

### 1. Clone the Repo
```bash
git clone https://github.com/arya251223/student_QA_Chatbot.git
cd student-support-chatbot
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv my_env
# For Linux/Mac
source my_env/bin/activate
# For Windows
my_env\Scripts\activate

pip install -r requirements.txt
```

### 3. Add Environment Variables
Create a `.env` file:
```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
ADMIN_USER_ID=your_numeric_admin_id
```

### 4. Run the Bot
```bash
python chatbot.py
```

---

## 🔑 Admin Commands
- `/teach` → Add new Q&A dynamically  
  Format:  
  ```
  /teach "What is the exam fee?" "fees" "The exam fee is ₹1000."
  ```
- `/stats` → View usage analytics  
- `/help` → See available commands  

---

## 📦 Dependencies
Listed in `requirements.txt`:

```
pandas>=2.0
python-telegram-bot>=20.8
python-dotenv>=1.0.1
sentence-transformers>=2.6.0
torch>=2.2.0
pyspellchecker==0.8.1
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🛠️ Example Usage

**Student:**  
`When is the last date for exam registration?`  

**Bot:**  
`The last date for exam registration is 15th March.`  

**Unknown Query:**  
`Can I bring my pet to hostel?`  

**Bot:**  
`I’m not fully sure about that. I’ll note this and improve later. Please rephrase your question.`  
*(also logged in `unanswered.csv`)*

---

## 🚨 Troubleshooting
- **Bot not starting?**  
  → Check `.env` for correct `TELEGRAM_BOT_TOKEN`.  

- **Always fallback response?**  
  → Ensure dataset is not empty and embeddings are rebuilt after `/teach`.  

- **Spelling correction misbehaving?**  
  → Add domain-specific words to `DOMAIN_WORDS` in `chatbot.py`.  

---

## 👨‍💻 Contributors
- **Aryan Kamble**  
- **Rajaj.tech**

---

## 📜MIT License
Copyright (c) 2025 **Aryan Kamble, Rajaj.tech**  

This software is provided **for internal, educational, and personal use only.**  


> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
