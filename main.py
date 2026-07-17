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

    target_folder =UPLOADS_DIR

    uploaded_files = st.file_uploader(
        "Chọn file để thêm",
        accept_multiple_files=True,
        key="file_uploader",
    )

    if st.button(" Lưu file vào thư mục"):
        if not uploaded_files:
            st.warning("Bạn chưa chọn file nào.")
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




user_question = st.chat_input("Nhập câu hỏi của bạn...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm tài liệu liên quan..."):
            gen = rag_answer_stream(user_question, emb)
            first_chunk = next(gen, None)

        def _chained_stream():
            if first_chunk is not None:
                yield first_chunk
            yield from gen

        answer = st.write_stream(_chained_stream())

    st.session_state.messages.append({"role": "assistant", "content": answer})