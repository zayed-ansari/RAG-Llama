from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.llms.ollama import Ollama
from langchain_classic.prompts import PromptTemplate
import requests
import os
# Testing it

def check_ollama_connection():
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

# Check connection first
print("Checking Ollama connection...")
if not check_ollama_connection():
    exit(1)

# Loading the PDF 
print("Loading the PDF...")
loader = PyPDFLoader("sample.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# Split into chunks 
print("Splitting document....")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks.")

# Create embeddings using Hugging Face 
print("Creating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("Vector store created")

# Setting up RAG chain with Ollama 
print("Setting up RAG Chain..")

try:
    llm = Ollama(
        model="llama3.2:latest", 
        base_url="http://localhost:11434",
        temperature=0
    )
    
    # Test the LLM connection with a simple prompt
    print("Testing LLM connection...")
    test_response = llm.invoke("Say 'Hello' in one word.")
    print(f"LLM connection test successful: {test_response}")
    
except Exception as e:
    print(f"Error setting up Ollama: {e}")
    exit(1)

# Providing a template
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

# Testing it
print("\n" + "="*50)
print("RAG system is Ready!")
print("="*50)

query = "What is the main topic of this document?"
try:
    result = qa_chain.invoke(query)
    print(f"\nQuery: {query}")
    print(f"\nAnswer: {result}")
except Exception as e:
    print(f"Error during query: {e}")
