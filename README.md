Function chatbot_TF_IDF is about an AI chatbox that answers academic questions by vectorizing the question 
in the database and the user's question, finding the sentence whose vectors are the closest, and answering 
the question. The LOSS parameter between the two vectors I use is 0.7

install libary:

pip install pandas scikit-learn joblib nltk numpy

run files in order TF_IDF_train.ipynb --> Interface_1.py



Function Chatbot_AI_Rag_LLM_qwen_ollamaws is also about ai chatbox that anwers academic questions following the
infomation in database, when find 5 most closest quention in database, we provide these infomation with Qwen AI in
OLLAMA an get the answer.

Pipline:
rag_train.ipynb                 Lmd_Utils.py           Interface_1.py
(Train 1 lần)                   (Xử lý logic)          (Giao diện)
      │                                │                       │
      ▼                                ▼                       ▼
 Load data                      Load models             Người dùng gõ câu hỏi
 (train.parquet                 (FAISS, embed,               │
  validation.parquet)           answers, questions)          ▼
      │                                                 _on_send() nhận input
      ▼                                                         │
 Embed câu hỏi                                                  ▼
 (vietnamese-sbert)                                     Thread riêng gọi
      │                                         Chatbot_AI_Rag_LLM_qwen_ollamaws()
      ▼                                                 │
 Lưu FAISS index                                        ▼
 + answers.pkl                          Find_closest_answer()
   + questions.pkl                       → FAISS tìm top 5 context
                                         (ngưỡng similarity = 0.7)
                                                       │
                                                      ▼
                                              Tìm thấy?
                                             ╱           ╲
                                           Có             Không
                                            │               │
                                            ▼               ▼
                                     Build_question()   "Xin lỗi tôi
                                     ghép context        không tìm thấy..."
                                     + câu hỏi
                                            │
                                            ▼
                                     Gửi prompt cho
                                     Qwen2.5:3b (Ollama)
                                            │
                                            ▼
                                     Sinh câu trả lời
                                     (streaming)
                                            │
                                            ▼
                                     Hiển thị lên UI
                                     (tkinter)
