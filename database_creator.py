import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS # Faster than Chroma

PDF_FILE_PATH = "data/Medical_book.pdf"
DB_PATH = "faiss_index" # Folder name change

def create_vector_db():
    print("📄 Loading PDF...")
    loader = PyPDFLoader(PDF_FILE_PATH)
    documents = loader.load()

    # Thode bade chunks rakhenge taaki context jaldi mile
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    print(f"🧩 Creating Index for {len(chunks)} chunks...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # FAISS index banana
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_PATH)
    print("🎉 Fast Database Saved!")

if __name__ == "__main__":
    create_vector_db()