import joblib
import nltk
import string
import re

import ollama

nltk.download('punkt')
nltk.download('punkt_tab')
all_word=[]
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
vectorizer = TfidfVectorizer()
stemmer = PorterStemmer()
import numpy as np
import re
import string
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer



EMBEDDING_MODEL  = 'keepitreal/vietnamese-sbert'
OLLAMA_MODEL     = 'qwen2.5:3b'
FAISS_INDEX_PATH = './models_rag/faiss_index.bin'
ANSWERS_PATH     = './models_rag/answers.pkl'
QUESTIONS_PATH   = './models_rag/questions.pkl'
simu=0.7


def bag_of_words(all_text,text):
    bag=np.zeros(len(all_text),dtype=np.float32)
    for idx, word in enumerate(all_text):
        if word in text:
            bag[idx]=1.0
    return bag

def cleantext(text):
    text = str(text.lower())
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
    text = re.sub(r".*\['(.*?)'\].*", r"\1", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text




def chatbot_TF_IDF(text,vectt,X_trainn):
    user_input=cleantext(text)
    user_vec = vectt.transform([user_input])
    sim = cosine_similarity(user_vec, X_trainn)
    return [sim.argmax()] if sim.max() >= simu else []






embed_model = SentenceTransformer(EMBEDDING_MODEL)
answers=joblib.load(ANSWERS_PATH)
questions=joblib.load(QUESTIONS_PATH)
faiss_index = faiss.read_index(FAISS_INDEX_PATH)

def Find_closest_answer(text,faiss_index,embed_model, top=5):
    vec = embed_model.encode([text], convert_to_numpy=True)
    faiss.normalize_L2(vec)
    scores, indices = faiss_index.search(vec, top)
    return [indices[0][i] for i in range(top) if scores[0][i]>=simu]


def Build_question(index,question):
    inf=""
    for i in index:
        inf+=answers[i]
        inf+="\n"
    full_text=f"""Bạn là trợ lý AI thông minh, luôn trả lời bằng tiếng Việt, ngắn gọn và chính xác.
Hãy dựa vào các thông tin tham khảo dưới đây để trả lời câu hỏi.
Nếu thông tin tham khảo không liên quan, hãy trả lời theo hiểu biết của bạn.
KHÔNG bịa đặt thông tin. KHÔNG lặp lại câu hỏi.

=== THÔNG TIN THAM KHẢO ===
{inf}

=== CÂU HỎI ===
{question}
"""
    return full_text


def Chatbot_AI_Rag_LLM_qwen_ollamaws(text):
    indx=Find_closest_answer(text,faiss_index,embed_model)
    for i in indx:
        print(i)
    if len(indx)!=0:
        full_text=Build_question(indx,text)
        resp=''
        for i in ollama.generate(model=OLLAMA_MODEL,prompt=full_text,stream=True):
            resp+= i.get('response')
    else:
        resp='Xin lỗi tôi không tìm thấy thông tin về câu hỏi của bạn trong database của tôi'
    return resp







