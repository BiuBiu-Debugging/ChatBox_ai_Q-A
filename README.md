# 🤖 RAG Document Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from company documents using semantic search with FAISS and Large Language Models (LLMs).

---

## 📌 Overview

RAG Document Assistant is an intelligent question-answering system designed to retrieve relevant information from technical documents before generating responses with an LLM.

Instead of relying only on the model's internal knowledge, the chatbot first searches a vector database for the most relevant document chunks, then uses those retrieved contexts to produce accurate and context-aware answers.

This project is suitable for:

* Enterprise document assistants
* Internal knowledge bases
* Technical documentation search
* PDF/Word/TXT document Q&A
* Customer support systems

---

## ✨ Features

* 📄 Read multiple document formats

  * PDF
  * Microsoft Word (.docx)
  * TXT

*  Automatic document chunking

*  Text preprocessing

*  Semantic search using Sentence Transformers

*  FAISS vector database for fast retrieval

*  Local LLM inference with Ollama

*  Interactive Streamlit web interface

*  Retrieval-Augmented Generation (RAG)

---



## 🛠 Technologies

* Python 3.12
* Streamlit
* FAISS
* Sentence Transformers
* Ollama
* LangChain (optional)
* PyPDF2
* python-docx
* NumPy

---



## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/BiuBiu-Debugging/ChatBox_ai_Q-A.git
cd ChatBox_ai_Q-A
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate

Linux

```bash
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Install Ollama

Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Download the language model

```bash
ollama pull qwen2.5:3b
```

Run Ollama

```bash
ollama serve
```

---

## ▶️ Run Application

```bash
streamlit run main.py
```

The application will start at

```
http://localhost:8501
```

---

## 🔍 Workflow

1. Load documents
2. Split documents into chunks
3. Generate embeddings
4. Store vectors in FAISS
5. User submits a question
6. Retrieve the most relevant chunks
7. Send retrieved context to Ollama
8. Generate the final answer

-

https://github.com/BiuBiu-Debugging
