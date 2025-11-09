# giveinfonew_bot_chatgpt_env_flex.py
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import numpy as np
import re
import os
from dotenv import load_dotenv

# ====== KHỞI ĐỘNG ======
print("🟢 [LOG] Đang khởi động NewsBot...")

# ====== ĐỌC FILE .env ======
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATA_PATH = os.getenv("DATA_PATH")

if not BOT_TOKEN:
    print("❌ [ERROR] Thiếu BOT_TOKEN trong file .env")
    raise SystemExit(1)

# ====== OPENAI KHÔNG BẮT BUỘC ======
client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ [LOG] Đã kết nối OpenAI.")
    except Exception as e:
        print("⚠️ [WARN] Không thể khởi tạo OpenAI client:", e)
else:
    print("ℹ️  [LOG] Không tìm thấy OPENAI_API_KEY — sẽ bỏ qua ChatGPT.")

# ====== ĐỌC DỮ LIỆU ======
try:
    print(f"📂 [LOG] Đọc dữ liệu từ: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    print(f"✅ [LOG] Đọc thành công CSV ({data.shape[0]} dòng).")
except Exception as e:
    print("❌ [LOG] Lỗi đọc CSV:", e)
    raise SystemExit(1)

# ====== LÀM SẠCH ======
def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

data["Title"] = data.get("Title", "").apply(clean_text)
data["Summary"] = data.get("Summary", "").apply(clean_text)
data["Contents"] = data.get("Contents", "").apply(clean_text)
data["combined_text"] = (data["Title"] + ". " + data["Summary"] + ". " + data["Contents"]).fillna("")

print("⚙️  [LOG] Khởi tạo TF-IDF...")
vectorizer = TfidfVectorizer(max_df=0.8, min_df=1, ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(data["combined_text"].values)
print("✅ [LOG] TF-IDF khởi tạo xong:", tfidf_matrix.shape)

# ====== HÀM TÌM BÀI ======
def find_similar_articles(query, top_n=3, min_score=0.15):
    q = clean_text(query)
    if not q:
        return []
    q_vec = vectorizer.transform([q])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_idx = np.argsort(sims)[::-1]
    results = []
    for idx in top_idx[:top_n]:
        score = float(sims[idx])
        if score >= min_score:
            row = data.iloc[idx]
            results.append({
                "score": score,
                "Title": row.get("Title", ""),
                "Summary": row.get("Summary", ""),
                "Contents": row.get("Contents", ""),
                "URL": row.get("URL", "")
            })
    return results

# ====== GỌI CHATGPT (NẾU CÓ) ======
def ask_chatgpt(prompt: str):
    if not client:
        return "⚠️ ChatGPT chưa được kích hoạt (thiếu OPENAI_API_KEY trong .env)."
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Bạn là NewsBot — trợ lý tiếng Việt thân thiện, biết trả lời tự nhiên."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ [LOG] Lỗi khi gọi OpenAI:", e)
        return "Xin lỗi, hiện tôi không thể truy cập ChatGPT."

# ====== Ý ĐỊNH CƠ BẢN ======
def detect_basic_intent(text):
    t = text.lower().strip()
    if any(g in t for g in ["hi", "hello", "chào", "xin chào", "hey"]):
        return "greeting"
    if any(g in t for g in ["cảm ơn", "thank", "thanks"]):
        return "thanks"
    if any(g in t for g in ["bạn tên gì", "tên bạn", "tên là gì"]):
        return "ask_name"
    if any(g in t for g in ["bạn khỏe", "khỏe không", "how are you"]):
        return "ask_how"
    if any(g in t for g in ["hôm nay", "ngày hôm nay", "today", "what day"]):
        return "ask_date"
    if t in ["/help", "help", "giúp", "hướng dẫn"]:
        return "help"
    return None

# ====== HANDLERS ======
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Xin chào 👋! Tôi là *NewsBot* — có thể giúp bạn tìm bài báo hoặc trả lời câu hỏi.\n"
        "Gõ một chủ đề (ví dụ: 'AI', 'giá vàng', 'bất động sản') để tôi tìm bài nhé!",
        parse_mode="Markdown"
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📘 Hướng dẫn:\n"
        "- Gõ chủ đề để tôi tìm bài báo.\n"
        "- Nếu không có, tôi sẽ nhờ ChatGPT (nếu được bật) hỗ trợ bạn.\n"
        "- Một số câu có sẵn: 'chào', 'bạn tên gì', 'hôm nay là ngày mấy'."
    )

async def reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text or ""
    print(f"💬 Người dùng: {user_text}")
    intent = detect_basic_intent(user_text)

    if intent == "greeting":
        await update.message.reply_text("Chào bạn! 😊 Tôi có thể giúp gì hôm nay?")
        return
    if intent == "thanks":
        await update.message.reply_text("Không có gì, rất vui được giúp bạn! 🤗")
        return
    if intent == "ask_name":
        await update.message.reply_text("Mình là NewsBot — trợ lý AI chuyên về tin tức và trả lời thông minh.")
        return
    if intent == "ask_how":
        await update.message.reply_text("Mình khỏe, cảm ơn bạn đã hỏi! 😄")
        return
    if intent == "ask_date":
        now = datetime.now().strftime("%A, %d %B %Y — %H:%M")
        await update.message.reply_text(f"Hôm nay là: {now}")
        return
    if intent == "help":
        await help_cmd(update, context)
        return

    # TÌM BÀI BÁO
    results = find_similar_articles(user_text, top_n=3, min_score=0.12)
    if results:
        for i, r in enumerate(results, start=1):
            title = r["Title"] or "Không có tiêu đề"
            summary = r["Summary"] or ""
            content = r["Contents"] or ""
            short = summary if summary else (content[:500] + "..." if len(content) > 500 else content)
            url = r["URL"] or ""
            msg = f"📰 *{i}. {title}*\n\n{short}\n\n🔗 [Đọc thêm]({url})"
            await update.message.reply_text(msg, parse_mode="Markdown", disable_web_page_preview=False)
        return

    # KHÔNG CÓ KẾT QUẢ → CHATGPT (NẾU CÓ)
    if client:
        print("🤖 Không có bài phù hợp → gọi ChatGPT...")
        gpt_reply = ask_chatgpt(user_text)
        await update.message.reply_text(gpt_reply)
    else:
        await update.message.reply_text(
            "Không tìm thấy bài báo nào phù hợp 😢.\n"
            "(ChatGPT chưa được bật — thêm `OPENAI_API_KEY` vào file .env để kích hoạt.)"
        )

# ====== MAIN ======
def main():
    print("🚀 Khởi động NewsBot...")
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply))
    print("✅ Bot đang chạy — chờ tin nhắn...")
    app.run_polling()

if __name__ == "__main__":
    main()
