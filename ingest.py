from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
import os

print("1. Loading file...")

file_path = "test_knowledge.txt" 
loader = TextLoader(file_path)
docs = loader.load()

print("2. Chunking the data for Mac M1 optimization...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  
    chunk_overlap=50 
)
chunks = text_splitter.split_documents(docs)
print(f"File split into {len(chunks)} chunks.")

print("3. Initializing Llama 3.2 Offline Embeddings...")
embeddings = OllamaEmbeddings(model="llama3.2")

print("4. Saving to local ChromaDB (Vector Memory)...")
persist_dir = "./doom_memory"

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_dir
)

print(f"Success! Knowledge is permanently saved in '{persist_dir}' directory.")
