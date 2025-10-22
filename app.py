import streamlit as st 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.llms.ollama import Ollama
from langchain_classic.prompts import PromptTemplate
import requests
import os
import tempfile

st.title("RAG document Q&A")
st.write("Upload a PDF and ask a question about it.")
def check_ollama_connection():
    """Check if Ollama is running and models are available"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            print("Available models:", [model['name'] for model in models])
            return True
        else:
            print(f"Ollama API returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Failed to connect to Ollama: {e}")
        return False
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Processing PDF......"):
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200
        )
        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        llm = Ollama(
        model="llama3.2:latest",
        base_url="http://localhost:11434",
        temperature=0
    )
        
        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Context: {context}
        Question: {question}

        Answer based on the context above:"""
        PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
        qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        chain_type_kwargs={"prompt":PROMPT},
                        retriever=vector_store.as_retriever(search_kwargs={"k": 5})
                        )
        st.session_state.qa_chain = qa_chain
        st.session_state.processed = True
    
    st.success(f"Processed {len(documents)} pages")
    os.unlink(tmp_path)

if "processed" in st.session_state:
    question = st.text_input("Your Question: ")

    if st.button("Get Answer") and question:
        with st.spinner("Thinking...."):
            result = st.session_state.qa_chain.invoke(question)
            st.write("****Answer:****")
            st.write(result['result']) 