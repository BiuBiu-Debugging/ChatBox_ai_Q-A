
import os
import time

from model.LLM import *
from model.register_file import *
import streamlit as st
from model.config import *
from model.document_embeding import  *
st.set_page_config(
    page_title="Trợ lý ảo Chatbot",
    page_icon="",
    layout="centered",
)

init_db()

@st.cache_resource
def load_embedding():
    return embedding()
emb = load_embedding()

st.title(" Trợ lý ảo Chatbot")
st.caption("Nhập câu hỏi bên dưới, trợ lý sẽ trả lời cho bạn.")



def get_answer(question: str) -> str:
    time.sleep(0.5)
    return f"Đây là câu trả lời mẫu cho câu hỏi: \"{question}\". Hãy thay hàm get_answer() bằng RAG pipeline thật của bạn."


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

with st.sidebar:
    st.header("Cài đặt")
    if st.button("🗑️ Xóa lịch sử hội thoại"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.subheader(" Thêm file vào thư mục")

    target_folder = st.text_input(
        "Đường dẫn thư mục lưu file",
        value=UPLOADS_DIR,
        help="File sẽ được lưu vào thư mục này nếu tên file chưa tồn tại trong database.",
    )

    uploaded_files = st.file_uploader(
        "Chọn file để thêm",
        accept_multiple_files=True,
        key="file_uploader",
    )

    if st.button(" Lưu file vào thư mục"):
        if not uploaded_files:
            st.warning("Bạn chưa chọn file nào.")
        elif not target_folder.strip():
            st.warning("Vui lòng nhập đường dẫn thư mục.")
        else:
            os.makedirs(target_folder, exist_ok=True)
            saved_count = 0
            skipped_count = 0

            for uf in uploaded_files:
                f_hash = hash_filename(uf.name)

                if file_hash_exists(f_hash):
                    skipped_count += 1
                    st.info(f" Bỏ qua (đã tồn tại): {uf.name}")
                    continue

                save_path = os.path.join(target_folder, uf.name)
                with open(save_path, "wb") as f:
                    f.write(uf.getbuffer())

                register_file(f_hash, uf.name, save_path)
                emb.add_document(save_path)
                saved_count += 1
                st.success(f" Đã lưu: {uf.name}")

            st.caption(f"Hoàn tất: {saved_count} file đã lưu, {skipped_count} file bị bỏ qua (trùng).")

    st.markdown("---")
    st.markdown(
        "**Hướng dẫn:**\n"
        "1. Nhập câu hỏi ở ô chat bên dưới.\n"
        "2. Nhấn Enter để gửi.\n"
        "3. Trợ lý sẽ hiển thị câu trả lời.\n\n"
        "Thay hàm `get_answer()` trong code để kết nối với RAG pipeline thật."
    )


user_question = st.chat_input("Nhập câu hỏi của bạn...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ..."):
            answer = rag_answer(user_question, emb)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})