from model.document_embeding import *

# llm.py
# Handles talking to the Ollama LLM and getting answers back.
#
# Uses proper system/human message separation so the model understands
# its role clearly. The prompt encourages the model to reason, analyze,
# and infer — not just parrot back chunks verbatim.
#
# The num_ctx parameter is set high enough to handle the context chunks
# we send (8 chunks × 1000 chars ≈ 2500 tokens) plus the prompt overhead.

import re

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

from model.config import LLM_MODEL, OLLAMA_BASE_URL


SYSTEM_PROMPT = """Bạn là một trợ lý AI thông minh chuyên hỗ trợ xử lý tài liệu. Nhiệm vụ của bạn là giúp người dùng hiểu, phân tích và khai thác thông tin từ các tài liệu mà họ đã tải lên.\n

Khả năng của bạn:\
- Trả lời trực tiếp các câu hỏi dựa trên ngữ cảnh của tài liệu được cung cấp.
- Phân tích và suy luận từ thông tin trong tài liệu (ví dụ: tính toán ngày tháng, so sánh dữ liệu, xác định xu hướng, nhận diện mẫu hoặc đưa ra các suy luận logic).
- Tóm tắt, giải thích và diễn giải nội dung của tài liệu.
- Nếu câu trả lời không được nêu trực tiếp trong tài liệu nhưng có thể suy luận hợp lý từ dữ liệu hiện có, hãy đưa ra kết luận và giải thích rõ quá trình suy luận của bạn.
- Nếu tài liệu thực sự không chứa đủ thông tin để trả lời, hãy trả lời một cách trung thực rằng không có đủ dữ liệu.

Quy tắc:
- Mọi câu trả lời phải dựa trên nội dung của tài liệu, không được tự ý thêm hoặc bịa đặt thông tin không được hỗ trợ bởi văn bản.
- Khi đưa ra suy luận, hãy nêu rõ rằng đó là suy luận chứ không phải thông tin được đề cập trực tiếp trong tài liệu.
- Luôn trả lời bằng cùng ngôn ngữ với câu hỏi của người dùng.
- QUAN TRỌNG: Bạn BẮT BUỘC phải trích dẫn nguồn. Ngữ cảnh được cung cấp bao gồm các đoạn văn (chunk) được đánh dấu theo dạng [Source X: filename]. Mỗi khi sử dụng thông tin từ một đoạn, hãy thêm ký hiệu trích dẫn dạng [^X] vào cuối câu tương ứng (trong đó X là số của nguồn).
- Không được hiển thị trực tiếp chuỗi "[Source X: filename]" trong câu trả lời. Chỉ sử dụng các ký hiệu trích dẫn dạng [^X]."""


USER_PROMPT_TEMPLATE = """\
Dưới đây là ngữ cảnh liên quan được trích xuất từ các tài liệu mà người dùng đã tải lên:

---
{context}
---

Question: {question}"""


NO_CONTEXT_PROMPT = """\
Người dùng đã đặt một câu hỏi nhưng hiện chưa có tài liệu nào được tải lên hoặc không tìm thấy tài liệu liên quan.\

Hãy thông báo rằng họ cần tải lên tài liệu trước để bạn có thể hỗ trợ, đồng thời hướng dẫn một cách thân thiện và hữu ích.


Question: {question}"""


_SOURCE_LABEL_RE = re.compile(
    r"\[Source\s+\d+\s*:\s*[^\]]*\]",
    re.IGNORECASE,
)


def _strip_source_labels(text: str) -> str:
    return _SOURCE_LABEL_RE.sub("", text)


def _clean_response(text: str) -> str:
    """
    Extract the actual answer from a model response, handling <think> tags.

    Strategy:
    1. If there's content AFTER the </think> closing tag, use that.
    2. If stripping the tags leaves nothing, extract from INSIDE the tags.
    3. If there's an unclosed <think> tag, grab what came after it.
    4. Fallback: strip all tags and return whatever's left.
    """
    if not text or not text.strip():
        return ""

    if "<think>" not in text:
        return text.strip()

    after_think = re.split(r"</think>", text, flags=re.IGNORECASE)
    if len(after_think) > 1:
        answer = after_think[-1].strip()
        if answer:
            return answer

    think_blocks = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if think_blocks:
        inner = think_blocks[-1].strip()
        if inner:
            return inner

    after_open = re.split(r"<think>", text, flags=re.IGNORECASE)
    if len(after_open) > 1:
        inner = after_open[-1].strip()
        if inner:
            return inner

    return re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()


def get_llm(model_name: str) -> ChatOllama:
    return ChatOllama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
        num_predict=2048,
        num_ctx=8192,  # enough for 8 chunks + prompt overhead
    )


def generate_answer(context: str, question: str, model_name: str = LLM_MODEL) -> str:

    llm = get_llm(model_name)

    system_content = SYSTEM_PROMPT

    if "qwen3" in model_name.lower():
        system_content += "\n\n/no_think"

    if context:
        user_content = USER_PROMPT_TEMPLATE.format(
            context=context, question=question,
        )
    else:
        user_content = NO_CONTEXT_PROMPT.format(question=question)

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content),
    ]

    response = llm.invoke(messages)
    answer = _strip_source_labels(_clean_response(response.content))

    if not answer:
        answer = (
            "Đã tìm thấy thông tin liên quan trong tài liệu của bạn, nhưng mô hình "
"không thể tạo ra một câu trả lời rõ ràng. Điều này đôi khi xảy ra với "
"các mô hình nhỏ. Hãy thử diễn đạt lại câu hỏi của bạn hoặc chuyển sang "
"một mô hình khác (ví dụ: qwen2.5)."
        )

    return answer

def rag_answer(question, emb: embedding):

    results = emb.search(
        question,
        top_k=4
    )

    if not results:
        return "Không tìm thấy thông tin trong tài liệu."


    context = ""

    for i, result in enumerate(results):

        context += (
            f"[Source {i+1}: {result['doc_id']}]\n"
            f"{result['text']}\n\n"
        )


    answer = generate_answer(
        context=context,
        question=question
    )

    return answer

def generate_answer_stream(context: str, question: str, model_name: str = LLM_MODEL):
    llm = get_llm(model_name)

    system_content = SYSTEM_PROMPT

    if "qwen3" in model_name.lower():
        system_content += "\n\n/no_think"

    if context:
        user_content = USER_PROMPT_TEMPLATE.format(
            context=context, question=question,
        )
    else:
        user_content = NO_CONTEXT_PROMPT.format(question=question)

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content),
    ]

    buffer = ""
    for chunk in llm.stream(messages):
        if chunk.content:
            buffer += chunk.content
            last_open = buffer.rfind("[")
            if last_open != -1 and buffer.find("]", last_open) == -1:
                safe = buffer[:last_open]
                if safe:
                    yield _strip_source_labels(safe)
                buffer = buffer[last_open:]
            else:
                yield _strip_source_labels(buffer)
                buffer = ""
    if buffer:
        yield _strip_source_labels(buffer)
