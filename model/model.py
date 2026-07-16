from langchain_ollama import OllamaEmbeddings
from model.config import *


def get_embeding_model():

    print("MODEL =", EMBEDDING_MODEL)
    print("URL =", OLLAMA_BASE_URL)

    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
models = get_embeding_model()
result = models.embed_query("Xin chào")

print(type(result))
print(len(result))
print(result[:5])