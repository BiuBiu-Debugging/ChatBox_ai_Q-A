import joblib
import nltk
import string
import re
nltk.download('punkt')
nltk.download('punkt_tab')
all_word=[]
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
vectorizer = TfidfVectorizer()
stemmer = PorterStemmer()
import numpy as np
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
import string

loss=0.8
stopwords = set(stopwords.words('english'))


def bag_of_words(all_text,text):
    bag=np.zeros(len(all_text),dtype=np.float32)
    for idx, word in enumerate(all_text):
        if word in text:
            bag[idx]=1.0
    return bag

def cleantext(text):
    text = str(text)
    text = re.sub(r".*\['(.*?)'\].*", r"\1", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text



def chatbot(user_input,vectorizern,X_trainn,answersn):
    user_input=cleantext(user_input)
    user_vec = vectorizern.transform([user_input])
    sim = cosine_similarity(user_vec, X_trainn)
    if (sim.max() > loss):
        idx = sim.argmax()
        return [answersn[idx], 1]
    else:
        return [
            'Sorry, I dont have the data to answer your question. I will forward your question to the support staff, please wait.',
            0]
vct="./models/tfidf_QA_VN.pkl"
ques="./models/Question_QA_VN.pkl"
aws="./models/Answer_QA_VN.pkl"
vector = joblib.load((vct))
model = joblib.load((ques))
answers = joblib.load((aws))

a="Phạm văn đồng là ai"
print(chatbot(a,vector,model,answers))



