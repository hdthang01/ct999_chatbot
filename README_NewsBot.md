
# 📰 NewsBot — Chatbot tra cứu bài báo thông minh

NewsBot là một chatbot Telegram có khả năng:
- Tìm kiếm và tóm tắt bài báo theo chủ đề người dùng nhập (VD: “giá vàng”, “bất động sản”, “COVID-19”...).
- Giao tiếp cơ bản: chào hỏi, cảm ơn, hỏi ngày giờ, tên bot,...
- Sử dụng TF-IDF và Cosine Similarity để tìm các bài báo liên quan trong file CSV huấn luyện.

---
# nguồn dataset
https://www.kaggle.com/datasets/trvminh/vietnamese-news-data/data
Vietnamese News Data

## ⚙️ 1. Cấu trúc dự án

```
📁 NewsBot/
├── giveinfonew_bot_smart.py     # Mã nguồn chính của bot
├── Dataset_articles.csv         # Dữ liệu bài báo huấn luyện
├── requirements.txt             # Danh sách thư viện cần cài
└── README.md                    # Hướng dẫn này
```

---

## 💻 2. Cài đặt môi trường

### Bước 1: Cài Python
- Cài Python 3.10+ từ: https://www.python.org/downloads/  
- Kiểm tra:
  ```bash
  python --version
  ```

### Bước 2: Cài thư viện cần thiết
Tạo file `requirements.txt` với nội dung:
```txt
python-telegram-bot==20.3
pandas
scikit-learn
numpy
```

Cài đặt bằng lệnh:
```bash
pip install -r requirements.txt
```

---

## 📦 3. Chuẩn bị dữ liệu huấn luyện

File `Dataset_articles.csv` cần có cấu trúc như sau:

| Title | Summary | Contents | URL |
|-------|----------|-----------|-----|
| Giá vàng hôm nay tăng nhẹ | Thị trường vàng biến động... | Nội dung chi tiết... | https://example.com/article1 |
| Kinh tế Việt Nam phục hồi | GDP quý 3 tăng... | Nội dung chi tiết... | https://example.com/article2 |

> ⚠️ Lưu ý:
> - File phải mã hóa UTF-8.
> - Nếu file của bạn có tên khác hoặc vị trí khác, sửa lại biến `DATA_PATH` trong mã nguồn.

---

## 🤖 4. Lấy token Telegram

1. Trên Telegram, tìm bot **@BotFather**.  
2. Gõ lệnh:
   ```
   /newbot
   ```
3. Đặt tên bot và username (phải kết thúc bằng `_bot`).  
4. BotFather sẽ gửi cho bạn một chuỗi token dạng:
   ```
   8278247397:AAHjnIM4772UO3dd6zPQhVrEdtiSF6Ezn48
   ```
5. Sao chép token đó và dán vào biến `BOT_TOKEN` trong file `giveinfonew_bot_smart.py`.

---

## 🚀 5. Chạy bot

Chạy lệnh trong thư mục chứa file:

```bash
python giveinfonew_bot_smart.py
```

Nếu thành công, bạn sẽ thấy log tương tự:

```
🟢 [LOG] Bắt đầu khởi động bot...
✅ [LOG] TF-IDF ma trận khởi tạo thành công
🚀 [LOG] Khởi động NewsBot Telegram...
🤖 [LOG] Bot đang chạy — chờ tin nhắn người dùng...
```

👉 Khi thấy dòng cuối cùng, bot **đã sẵn sàng**.  
Mở Telegram, tìm **username bot** bạn đã tạo, và bắt đầu chat.

---

## 💬 6. Một số câu bạn có thể thử

| Câu hỏi / Lệnh | Kết quả dự kiến |
|----------------|----------------|
| `chào` | “Chào bạn! Mình có thể giúp gì hôm nay?” |
| `bạn tên gì` | “Mình là NewsBot — bot tìm bài báo.” |
| `hôm nay là ngày mấy` | Bot trả về ngày giờ hiện tại |
| `giá vàng` | Bot gửi danh sách các bài báo liên quan |
| `/help` | Hướng dẫn cách sử dụng bot |

---

## 🧠 7. Cách hoạt động

1. Bot đọc dữ liệu từ file CSV.  
2. Làm sạch văn bản (`clean_text`), ghép các cột `Title`, `Summary`, `Contents`.  
3. Sử dụng `TfidfVectorizer` để biểu diễn văn bản thành vector.  
4. Khi người dùng gửi câu hỏi:
   - Nếu là **câu trò chuyện**, bot phản hồi tự nhiên.
   - Nếu là **chủ đề**, bot tính **cosine similarity** và trả về 3 bài báo gần nhất.

---

## 🧩 8. Khắc phục lỗi thường gặp

| Lỗi | Nguyên nhân | Cách khắc phục |
|------|--------------|----------------|
| `InvalidToken` | Token sai hoặc hết hạn | Lấy token mới từ @BotFather |
| `pd not defined` | Quên import pandas | Thêm `import pandas as pd` |
| Không in ra log | Chưa bật logging hoặc lỗi sớm | Thêm `print()` hoặc `logging.debug()` |
| Không chat được | Chưa thấy dòng “Bot đang chạy...” | Kiểm tra token & kết nối Internet |
| Không tìm thấy bài | Dữ liệu ít hoặc từ khóa lạ | Giảm `min_score` trong `find_similar_articles()` |

---

## 🧾 9. Bản quyền & Tác giả

- **Tác giả:** NLTANH  
- **Mục đích:** Dự án học tập, nghiên cứu năng lực số và ứng dụng NLP tiếng Việt.  
- **Phiên bản:** 1.0 — Tháng 10/2025  

---
