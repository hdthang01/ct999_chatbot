# giveinfonew_bot_smart.py
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import numpy as np
import re

print("🟢 [LOG] Bắt đầu khởi động bot...")

# ====== THAY TOKEN Ở ĐÂY ======
BOT_TOKEN = "8278247397:AAHjnIM4772UO3dd6zPQhVrEdtiSF6Ezn48"

# ====== ĐỌC DỮ LIỆU VÀ CHUẨN HÓA ======
try:
    DATA_PATH = "D:/TaiLieuThacSi/NangLucSo/code/Dataset_articles.csv"
    print(f"📂 [LOG] Đang đọc dữ liệu từ: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    print(f"✅ [LOG] Đọc thành công CSV: {data.shape[0]} dòng, {data.shape[1]} cột")
except Exception as e:
    print("❌ [LOG] Lỗi khi đọc file CSV:", e)
    raise SystemExit(1)

# Tạo 1 cột text kết hợp để vectorize (Title + Summary + Contents)
def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

print("🔧 [LOG] Đang làm sạch dữ liệu văn bản...")
data["Title"] = data.get("Title", "").apply(clean_text)
data["Summary"] = data.get("Summary", "").apply(clean_text)
data["Contents"] = data.get("Contents", "").apply(clean_text)
data["combined_text"] = (data["Title"] + ". " + data["Summary"] + ". " + data["Contents"]).fillna("")
print("✅ [LOG] Hoàn tất làm sạch dữ liệu.")

# Build TF-IDF vectorizer and matrix once at startup
print("⚙️ [LOG] Đang tạo TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(max_df=0.8, min_df=1, ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(data["combined_text"].values)
print("✅ [LOG] TF-IDF ma trận khởi tạo thành công:", tfidf_matrix.shape)

# ====== HÀM TÌM KIẾM BẰNG COSINE SIMILARITY ======
def find_similar_articles(query, top_n=3, min_score=0.15):
    print(f"🔍 [LOG] Bắt đầu tìm bài cho truy vấn: '{query}'")
    q = clean_text(query)
    if not q:
        print("⚠️ [LOG] Truy vấn rỗng.")
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
    print(f"✅ [LOG] Tìm thấy {len(results)} kết quả phù hợp (min_score={min_score})")
    return results

# ====== HÀM XỬ LÝ Ý ĐỊNH ======
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

# ====== HANDLERs BOT ======
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("📨 [LOG] Nhận lệnh /start từ người dùng.")
    await update.message.reply_text(
        "Xin chào 👋! Tôi là NewsBot — bạn có thể hỏi tôi về chủ đề để tìm bài báo.\n"
        "Ví dụ: gửi 'bất động sản', 'ngọc trinh', 'COVID-19'. Gõ 'help' để biết thêm."
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("📨 [LOG] Nhận lệnh /help.")
    await update.message.reply_text(
        "Hướng dẫn nhanh:\n"
        "- Gõ 1 chủ đề hoặc câu hỏi (ví dụ: 'giá vàng hôm nay', 'bất động sản') để tìm bài.\n"
        "- Một số câu thoại: 'chào', 'bạn tên gì', 'hôm nay là ngày mấy'.\n"
        "- Nếu muốn tắt preview link, dùng lệnh /nopreview"
    )

async def reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text or ""
    print(f"💬 [LOG] Người dùng gửi: {user_text}")
    intent = detect_basic_intent(user_text)
    print(f"🧠 [LOG] Ý định phát hiện: {intent}")

    # Xử lý intent cơ bản
    if intent == "greeting":
        await update.message.reply_text("Chào bạn! Mình có thể giúp gì hôm nay? 😊")
        return
    if intent == "thanks":
        await update.message.reply_text("Bạn rất hoan nghênh! Nếu cần tìm bài báo, cứ gửi chủ đề nhé.")
        return
    if intent == "ask_name":
        await update.message.reply_text("Mình là NewsBot — bot tìm và tóm tắt bài báo. Rất vui được giúp!")
        return
    if intent == "ask_how":
        await update.message.reply_text("Mình ổn! Cảm ơn bạn đã hỏi. Bạn cần tìm thông tin gì?")
        return
    if intent == "ask_date":
        now = datetime.now().strftime("%A, %d %B %Y — %H:%M")
        await update.message.reply_text(f"Hôm nay là: {now}")
        return
    if intent == "help":
        await help_cmd(update, context)
        return

    # Nếu không phải small-talk, tìm bài báo
    results = find_similar_articles(user_text, top_n=3, min_score=0.12)
    if not results:
        print("❌ [LOG] Không tìm thấy kết quả nào phù hợp.")
        await update.message.reply_text("Xin lỗi 😢, tôi không tìm thấy bài báo nào liên quan. Bạn thử cụm từ khác nhé.")
        return

    print(f"📦 [LOG] Đang gửi {len(results)} kết quả cho người dùng...")
    for i, r in enumerate(results, start=1):
        title = r["Title"] or "No title"
        summary = r["Summary"] or ""
        content = r["Contents"] or ""
        short = summary if summary else (content[:500] + "..." if len(content) > 500 else content)
        url = r["URL"] or ""
        msg = f"🔎 *{i}. {title}*\n\n{short}\n\n🔗 [Đọc tiếp]({url})"
        await update.message.reply_text(msg, parse_mode="Markdown", disable_web_page_preview=False)

# ====== KHỞI ĐỘNG BOT ======
def main():
    print("🚀 [LOG] Khởi động NewsBot Telegram...")
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply))
    print("🤖 [LOG] Bot đang chạy — chờ tin nhắn người dùng...")
    app.run_polling()

if __name__ == "__main__":
    main()
