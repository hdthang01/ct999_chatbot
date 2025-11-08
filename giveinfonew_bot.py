# giveinfonew_bot_semantic.py
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import pandas as pd
from datetime import datetime
import numpy as np
import re
import os
from dotenv import load_dotenv

# ==== NEW: sentence-transformers for semantic embeddings ====
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise SystemExit(
        "Thiếu thư viện 'sentence-transformers'. Cài đặt bằng:\n"
        "    pip install sentence-transformers torch --upgrade"
    )

# ====== KHỞI ĐỘNG ======
print("🟢 [LOG] Đang khởi động NewsBot (Semantic Embedding)...")

# ====== ĐỌC FILE .env ======
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATA_PATH = os.getenv("DATA_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")  # tốt cho tiếng Việt

if not BOT_TOKEN:
    print("❌ [ERROR] Thiếu BOT_TOKEN trong file .env")
    raise SystemExit(1)

# ====== OPENAI (KHÔNG BẮT BUỘC) ======
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

# Chuẩn hóa cột
data["Title"] = data.get("Title", "").apply(clean_text)
data["Summary"] = data.get("Summary", "").apply(clean_text)
data["Contents"] = data.get("Contents", "").apply(clean_text)

# Ghép văn bản
data["combined_text"] = (data["Title"] + ". " + data["Summary"] + ". " + data["Contents"]).fillna("").apply(clean_text)

# ====== EMBEDDING MODEL ======
print(f"⚙️  [LOG] Tải mô hình embedding: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)

# Encode toàn bộ corpus: normalize để cosine = dot
print("🔧 [LOG] Mã hoá embedding cho toàn bộ dữ liệu (có thể mất vài giây)...")
corpus_texts = data["combined_text"].tolist()
corpus_embeddings = model.encode(
    corpus_texts,
    batch_size=64,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)
print("✅ [LOG] Tạo xong corpus embeddings:", corpus_embeddings.shape)

# ====== HÀM TÌM BÀI (SEMANTIC) ======
def find_similar_articles(query, top_n=3, min_score=0.15):
    q = clean_text(query)
    if not q:
        return []

    # Encode câu hỏi (đã chuẩn hoá)
    query_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)  # (1, d)
    # Cosine = dot vì đã normalize
    sims = (query_emb @ corpus_embeddings.T).flatten()  # shape: (N,)

    top_idx = np.argsort(sims)[::-1]
    results = []
    picked = 0
    for idx in top_idx:
        score = float(sims[idx])
        if score < min_score:
            break
        row = data.iloc[idx]
        results.append({
            "score": score,
            "Title": row.get("Title", ""),
            "Summary": row.get("Summary", ""),
            "Contents": row.get("Contents", ""),
            "URL": row.get("URL", "")
        })
        picked += 1
        if picked >= top_n:
            break
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
        "Xin chào 👋! Tôi là *NewsBot* — đã nâng cấp Semantic Search.\n"
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

    # TÌM BÀI BÁO (SEMANTIC)
    results = find_similar_articles(user_text, top_n=3, min_score=0.12)
    if results:
        for i, r in enumerate(results, start=1):
            title = r["Title"] or "Không có tiêu đề"
            summary = r["Summary"] or ""
            content = r["Contents"] or ""
            short = summary if summary else (content[:500] + "..." if len(content) > 500 else content)
            url = r["URL"] or ""
            score = r["score"]
            msg = (
                f"📰 *{i}. {title}*\n"
                f"⭐ Độ tương đồng: `{score:.3f}`\n\n"
                f"{short}\n\n"
                f"{'🔗 [Đọc thêm](' + url + ')' if url else ''}"
            )
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
    print("🚀 Khởi động NewsBot (Semantic Embedding)...")
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply))
    print("✅ Bot đang chạy — chờ tin nhắn...")
    app.run_polling()

if __name__ == "__main__":
    main()
