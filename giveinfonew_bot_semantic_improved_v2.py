from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import pandas as pd
from datetime import datetime
import numpy as np
import re
import os
import json
from dotenv import load_dotenv

# semantic libs
from sentence_transformers import SentenceTransformer
import faiss

# ===== CONFIG =====
load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')
DATA_PATH = os.getenv('DATA_PATH', 'articles.csv')
# default switched to multilingual model for Vietnamese/other languages
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/distiluse-base-multilingual-cased-v2') # cái này ok nè hẹ hẹ
CHUNK_SIZE = int(os.getenv('EMB_CHUNK_SIZE', '20000'))   # 20k dòng mỗi chunk
BATCH_SIZE = int(os.getenv('EMB_BATCH_SIZE', '32'))
MIN_SCORE = float(os.getenv('MIN_SCORE', '0.12'))
TOP_K = int(os.getenv('TOP_K', '3'))  # how many candidates to retrieve from FAISS
EMB_DIR = os.getenv('EMB_DIR', './v2_embeds')
FAISS_INDEX_PATH = os.path.join(EMB_DIR, 'faiss_index.bin')
EMB_META_PATH = os.path.join(EMB_DIR, 'emb_meta.json')

os.makedirs(EMB_DIR, exist_ok=True)

if not BOT_TOKEN:
    raise SystemExit('Missing BOT_TOKEN in .env')

# ===== read data =====
print('[LOG] Reading data from', DATA_PATH)
try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    raise SystemExit(f'Cannot read DATA_PATH={DATA_PATH}: {e}')

# normalize columns
for c in ['Title','Summary','Contents']:
    if c not in df.columns:
        df[c] = ''

# simple cleaner
def clean_text(s):
    if pd.isna(s):
        return ''
    t = str(s)
    t = re.sub(r'<[^>]+>', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

for c in ['Title','Summary','Contents']:
    df[c] = df[c].map(clean_text)

df['combined_text'] = (df['Title'] + '. ' + df['Summary'] + '. ' + df['Contents']).map(clean_text)
texts = df['combined_text'].fillna('').tolist()
N = len(texts)
print(f'[LOG] {N} documents')

# ===== model init =====
# auto detect device 
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[LOG] Loading embedding model {EMBEDDING_MODEL} on {DEVICE}')
# instantiate model
model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)

if DEVICE == 'cpu':
    import os
    ncpu = os.cpu_count() or 1
    torch.set_num_threads(ncpu)
    os.environ['OMP_NUM_THREADS'] = str(ncpu)
    os.environ['MKL_NUM_THREADS'] = str(ncpu)

# ===== chunked encoding =====
part_files = {}
for start in range(0, N, CHUNK_SIZE):
    fname = os.path.join(EMB_DIR, f'emb_part_{start}.npy')
    part_files[start] = fname

# if meta exists, validate
meta = {}
if os.path.exists(EMB_META_PATH):
    try:
        with open(EMB_META_PATH, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except Exception:
        meta = {}

need_encode_parts = []
for start, fname in part_files.items():
    if os.path.exists(fname):
        print(f'[LOG] found part file {fname} -> will skip encode for this range')
    else:
        need_encode_parts.append((start, fname))

# quick helper to encode a list

def encode_texts(text_list, batch_size=BATCH_SIZE):
    # returns normalized float32 numpy array
    emb = model.encode(text_list, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    return emb.astype('float32')

# encode missing parts
for start, fname in need_encode_parts:
    end = min(start + CHUNK_SIZE, N)
    subtexts = texts[start:end]
    if len(subtexts) == 0:
        continue
    print(f'[LOG] Encoding part {start}:{end} -> saving to {fname}')
    emb = encode_texts(subtexts)
    np.save(fname, emb)
    print(f'[LOG] Saved {fname} shape={emb.shape}')

# load and stack parts (keeping original order)
arrays = []
loaded = 0
for start in range(0, N, CHUNK_SIZE):
    fname = part_files[start]
    if not os.path.exists(fname):
        raise SystemExit(f'Missing embedding part {fname} - resume failed')
    a = np.load(fname)
    arrays.append(a)
    loaded += a.shape[0]

corpus_embeddings = np.vstack(arrays)
# if corpus_embeddings.shape[0] != N:
    # raise SystemExit(f'Embedding count mismatch {corpus_embeddings.shape[0]} != {N}')

D = corpus_embeddings.shape[1]
print(f'[LOG] Corpus embeddings loaded shape = {corpus_embeddings.shape}, dim={D}')

# write metadata
meta = dict(model=EMBEDDING_MODEL, n=N, dim=D, normalized=True)
with open(EMB_META_PATH, 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

# ===== build or load FAISS index =====n
if os.path.exists(FAISS_INDEX_PATH):
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        print('[LOG] Loaded FAISS index from', FAISS_INDEX_PATH)
    except Exception as e:
        print('[WARN] Failed to load FAISS index, rebuilding. error=', e)
        index = faiss.IndexFlatIP(D)
        index.add(corpus_embeddings)
        faiss.write_index(index, FAISS_INDEX_PATH)
else:
    index = faiss.IndexFlatIP(D)
    index.add(corpus_embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print('[LOG] Built FAISS index and saved to', FAISS_INDEX_PATH)

# ===== search function =====

def find_similar_articles(query, top_k=3, min_score=MIN_SCORE):
    q = clean_text(query)
    if not q:
        return []
    q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
    # faiss search
    scores, indices = index.search(q_emb, TOP_K)
    scores = scores.flatten()
    indices = indices.flatten()
    results = []
    for sc, idx in zip(scores, indices):
        if idx < 0 or sc < min_score:
            continue
        row = df.iloc[idx]
        results.append({
            'score': float(sc),
            'Title': row.get('Title',''),
            'Summary': row.get('Summary',''),
            'Contents': row.get('Contents',''),
            'URL': row.get('URL','')
        })
    return results

# ===== simple intent & chatGPT fallback =====
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print('[WARN] cannot init OpenAI client:', e)


def ask_chatgpt(prompt: str):
    if not client:
        return None
    try:
        r = client.chat.completions.create(model='gpt-4o-mini', messages=[{'role':'user','content':prompt}], max_tokens=400)
        return r.choices[0].message.content.strip()
    except Exception as e:
        print('[WARN] chatgpt error', e)
        return None


def detect_basic_intent(text):
    t = (text or '').lower()
    if any(g in t for g in ['chào','hi','hello','xin chào','hey']): return 'greeting'
    if any(g in t for g in ['cảm ơn','thanks','thank']): return 'thanks'
    if 'tên' in t and 'gì' in t: return 'ask_name'
    return None

# ===== telegram handlers =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Xin chào! Tôi là NewsBot — semantic search (multilingual). Gõ chủ đề để tìm bài.')

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Gõ 1 chủ đề, tôi sẽ tìm bài liên quan. Nếu không tìm được, tôi sẽ dùng ChatGPT (nếu bật).')

async def reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text or ''
    intent = detect_basic_intent(user_text)
    if intent == 'greeting':
        await update.message.reply_text('Chào bạn!')
        return
    if intent == 'thanks':
        await update.message.reply_text('Không có gì!')
        return

    results = find_similar_articles(user_text, top_k=3, min_score=MIN_SCORE)
    if results:
        # richer response: trả TOP_K gợi ý và tóm tắt 1 câu (summary nếu có)
        msgs = []
        for i, r in enumerate(results, start=1):
            title = r['Title'] or 'Không có tiêu đề'
            summ = r['Summary'] or (r['Contents'][:300] + '...' if r['Contents'] else '')
            url = r["URL"] or ""
            if url:
                msgs.append(f"*{i}. {title}* (score={r['score']:.3f})\n{summ}\n🔗 [Đọc thêm]({url})")
            else:
                msgs.append(f"*{i}. {title}* (score={r['score']:.3f})\n{summ}")

        await update.message.reply_text(
            '\n\n'.join(msgs),
            parse_mode='Markdown',
            disable_web_page_preview=False
        )

        return

    # fallback
    if client:
        g = ask_chatgpt(user_text)
        if g:
            await update.message.reply_text(g)
            return

    await update.message.reply_text('Không tìm thấy kết quả. Thử diễn đạt khác hoặc cung cấp thêm dữ liệu.')

# ===== main =====
def main():
    print('[LOG] Starting Telegram bot...')
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('help', help_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply))
    app.run_polling()

if __name__ == '__main__':
    main()
