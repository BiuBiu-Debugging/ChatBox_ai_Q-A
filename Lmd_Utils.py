import joblib
import nltk
import string
import re
nltk.download('punkt')
nltk.download('punkt_tab')
all_word=[]
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
vectorizer = TfidfVectorizer()
stemmer = PorterStemmer()
import numpy as np
from nltk.corpus import stopwords
import re
import string
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer



EMBEDDING_MODEL = 'keepitreal/vietnamese-sbert'
stopwords = set(stopwords.words('english'))
simu=0.8


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


def Chatbot_Rag_embed(text,index, top=1):
    vec = embed_model.encode([text], convert_to_numpy=True)
    faiss.normalize_L2(vec)
    scores, indices = index.search(vec, top)
    return [indices[0][i] for i in range(top) if scores[0][i]>=simu]


