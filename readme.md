# RAG-Llama 🦙

A **local Retrieval-Augmented Generation (RAG) system** with Llama 3.2 and Streamlit UI.  
Upload PDFs, ask questions in natural language, and get detailed answers with context — fully offline, no API costs.

---

## 🚀 Features

- Upload PDF documents
- Natural language Q&A
- Context-aware answers using RAG
- Powered by **Llama 3.2** (local)
- **No API keys or cloud required**
- Streamlit UI for interactive use

---

## 🧠 Tech Stack

- **LangChain** – for RAG pipeline and document retrieval  
- **ChromaDB** – vector database for embeddings  
- **Sentence Transformers** – to convert text into embeddings  
- **Ollama (Llama 3.2)** – local large language model  
- **Streamlit** – interactive frontend  
- **Python** – main programming language

---

## ⚙️ Quick Start

1. Install all the required libraries
```bash
pip install -r requirements.txt 
```
2. Pull Llama 3.2 model (for Ollama)
```bash
ollama pull llama3.2:latest
```
3. Run the Streamlit app
```bash
streamlit run app.py
```
