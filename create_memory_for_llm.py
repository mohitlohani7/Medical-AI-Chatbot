import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDFs.")
    return documents

def create_chunks(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks from documents.")
    return chunks

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_and_save_vectorstore(chunks, embedding_model, save_path):
    db = FAISS.from_documents(chunks, embedding_model)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    db.save_local(save_path)
    print(f"Vectorstore saved at {save_path}")

if __name__ == "__main__":
    docs = load_pdf_files(DATA_PATH)
    chunks = create_chunks(docs)
    embedding_model = get_embedding_model()
    create_and_save_vectorstore(chunks, embedding_model, DB_FAISS_PATH)
