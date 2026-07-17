# 🤖 RAG Document Assistant

Trợ lý ảo chatbot hỏi-đáp tài liệu, sử dụng kiến trúc **RAG (Retrieval-Augmented Generation)** với LLM chạy local qua **Ollama**. Người dùng tải tài liệu lên (PDF, DOCX, TXT), hệ thống sẽ trích xuất, chia nhỏ, embedding và lưu vào vector database. Khi đặt câu hỏi, hệ thống tìm các đoạn văn bản liên quan nhất và đưa vào LLM để sinh câu trả lời có trích dẫn nguồn.

## ✨ Tính năng chính

- **Giao diện chat** trực quan bằng [Streamlit](https://streamlit.io/), hỗ trợ tiếng Việt.
- **Upload & quản lý tài liệu**: hỗ trợ `.pdf`, `.docx`, `.txt`; tự động chống trùng lặp bằng hash tên file, lưu metadata vào SQLite.
- **Hybrid Search**: kết hợp tìm kiếm ngữ nghĩa (FAISS vector search) và tìm kiếm từ khóa (BM25), hợp nhất kết quả bằng thuật toán **Reciprocal Rank Fusion (RRF)**.
- **Re-ranking**: sử dụng Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) để sắp xếp lại kết quả theo độ liên quan trước khi đưa vào LLM.
- **Trích dẫn nguồn**: câu trả lời của LLM luôn đi kèm ký hiệu trích dẫn `[^X]` tương ứng với tài liệu nguồn.
- **LLM & Embedding chạy local** thông qua Ollama (mặc định: `qwen2.5` cho sinh câu trả lời, `nomic-embed-text` cho embedding).
- **Chunking thông minh** bằng `RecursiveCharacterTextSplitter` (LangChain), có thể tuỳ chỉnh kích thước & độ chồng lấp.
- Hỗ trợ **streaming** câu trả lời (sinh từng phần theo thời gian thực).

## 🏗️ Kiến trúc & luồng xử lý

```
┌─────────────┐     upload      ┌──────────────────┐
│   Người     │ ──────────────► │  register_file.py │──► SQLite (chống trùng file)
│   dùng      │                 └──────────────────┘
│ (Streamlit) │                          │
└─────────────┘                          ▼
      │                        ┌──────────────────────┐
      │                        │ document_embeding.py  │
      │                        │  - Đọc file (pdf/docx)│
      │  câu hỏi               │  - Chunking văn bản   │
      ▼                        │  - Embedding (Ollama) │
┌─────────────┐                │  - Lưu FAISS + BM25   │
│   LLM.py    │◄──── search ───│                       │
│ rag_answer()│                └──────────────────────┘
│  - Hybrid   │
│    search   │
│  - Rerank   │
│  - Sinh câu │
│    trả lời  │
└─────────────┘
```

**Luồng thêm tài liệu:**
1. Người dùng chọn file trong sidebar → hệ thống hash tên file, kiểm tra trùng trong SQLite.
2. Nếu chưa tồn tại: lưu file vào thư mục `data/uploads`, ghi metadata vào DB, sau đó chia nhỏ (chunk) và embedding, lưu vào FAISS index.

**Luồng hỏi-đáp (RAG):**
1. Câu hỏi được tìm kiếm song song bằng FAISS (vector similarity) và BM25 (keyword).
2. Kết quả được hợp nhất bằng RRF, sau đó re-rank bằng Cross-Encoder.
3. Top-K đoạn văn bản liên quan nhất được đưa vào prompt cùng câu hỏi.
4. LLM (qua Ollama) sinh câu trả lời kèm trích dẫn nguồn `[^X]`.

## 📁 Cấu trúc thư mục

```
RAG_DOCUMENT_ASSISTANT/
├── .venv/                      # Môi trường ảo Python
├── data/
│   ├── uploads/                # Tài liệu người dùng tải lên (pdf, docx, txt)
│   └── vector_db/               # Dữ liệu vector database
│       ├── faiss.index          # FAISS index (vector search)
│       ├── vectors.npy          # Ma trận vector embedding
│       ├── chunks.npy           # Danh sách đoạn văn bản đã chia nhỏ
│       └── doc_ids.npy          # ID tài liệu tương ứng với từng chunk
├── model/
│   ├── .env                     # Biến môi trường (không commit lên git)
│   ├── config.py                 # Cấu hình chung (đường dẫn, model, tham số)
│   ├── document_embeding.py      # Đọc, chunk, embedding & tìm kiếm tài liệu
│   ├── LLM.py                    # Giao tiếp với Ollama LLM, sinh câu trả lời RAG
│   ├── model.py                  # Khởi tạo embedding model
│   └── register_file.py          # Quản lý metadata file (SQLite)
├── file_uploads.db               # Cơ sở dữ liệu SQLite lưu thông tin file đã upload
├── main.py                       # Điểm khởi chạy ứng dụng Streamlit
└── README.md
```

## ⚙️ Yêu cầu hệ thống

- Python 3.10+
- [Ollama](https://ollama.com/) đã được cài đặt và đang chạy local
- Các model Ollama cần thiết:
  ```bash
  ollama pull qwen2.5
  ollama pull nomic-embed-text
  ```

## 🚀 Cài đặt

1. **Clone repo**

   ```bash
   git clone <repo-url>
   cd RAG_DOCUMENT_ASSISTANT
   ```

2. **Tạo môi trường ảo & cài dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Cấu hình biến môi trường**

   Tạo file `model/.env` với nội dung (tùy chỉnh theo nhu cầu):

   ```env
   OLLAMA_BASE_URL=http://localhost:11434
   LLM_MODEL=qwen2.5:latest
   EMBEDDING_MODEL=nomic-embed-text:latest
   TOP_K=8
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   MAX_FILE_SIZE_MB=50
   RATE_LIMIT=30/minute
   API_KEY=
   ```

4. **Khởi động Ollama** (nếu chưa chạy)

   ```bash
   ollama serve
   ```

5. **Chạy ứng dụng**

   ```bash
   streamlit run main.py
   ```

   Ứng dụng sẽ mở tại `http://localhost:8501`.

## 📝 Hướng dẫn sử dụng

1. Mở sidebar, chọn **"Thêm file vào thư mục"** → tải lên các file `.pdf`, `.docx` hoặc `.txt`.
2. Nhấn **"Lưu file vào thư mục"** để lưu và tự động embedding tài liệu.
3. Nhập câu hỏi vào ô chat ở cuối trang, nhấn Enter.
4. Trợ lý sẽ tìm các đoạn tài liệu liên quan và trả lời kèm trích dẫn nguồn.
5. Có thể xóa lịch sử hội thoại bằng nút **"🗑️ Xóa lịch sử hội thoại"** trong sidebar.

## 🔧 Cấu hình nâng cao

| Biến môi trường     | Mô tả                                              | Mặc định                  |
|----------------------|-----------------------------------------------------|----------------------------|
| `OLLAMA_BASE_URL`    | Địa chỉ Ollama server                                | `http://localhost:11434`  |
| `LLM_MODEL`          | Model dùng để sinh câu trả lời                       | `qwen2.5:latest`          |
| `EMBEDDING_MODEL`    | Model dùng để embedding văn bản                      | `nomic-embed-text:latest` |
| `TOP_K`              | Số đoạn văn bản liên quan nhất được lấy khi truy vấn | `8`                        |
| `CHUNK_SIZE`         | Kích thước mỗi đoạn văn bản (ký tự)                  | `1000`                     |
| `CHUNK_OVERLAP`      | Độ chồng lấp giữa các đoạn                           | `200`                      |
| `MAX_FILE_SIZE_MB`   | Giới hạn dung lượng file upload                      | `50`                       |
| `RATE_LIMIT`         | Giới hạn tần suất request                            | `30/minute`                |

## 📦 Công nghệ sử dụng

- **Streamlit** – giao diện chat
- **LangChain** (`langchain-ollama`, `langchain-text-splitters`) – tích hợp LLM & chunking
- **FAISS** – vector database cho tìm kiếm ngữ nghĩa
- **rank_bm25** – tìm kiếm từ khóa (BM25)
- **sentence-transformers (CrossEncoder)** – re-ranking kết quả tìm kiếm
- **pypdf**, **python-docx** – trích xuất nội dung tài liệu
- **SQLite** – lưu trữ metadata file đã upload
- **Ollama** – chạy LLM & embedding model local





