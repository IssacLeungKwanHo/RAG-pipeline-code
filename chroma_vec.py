import logging
import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

def create_vector_store():
    BASE_DIR = "/Users/issacleung/Desktop/RAGpipline"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
    
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_kwargs={"device": "cpu", "trust_remote_code": True}
    )

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        try:
            vector_db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model, client=chroma_client)
            logging.info("Loaded existing Chroma vector store")
            return vector_db
        except:
            pass

    # Build new store
    txt_files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No text files in {DATA_DIR}")

    documents = []
    for txt_file in txt_files:
        loader = TextLoader(txt_file, encoding="utf-8")
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR,
        client=chroma_client
    )
    logging.info(f"Created new vector store with {len(chunks)} chunks")
    return vector_db

if __name__ == "__main__":
    create_vector_store()
    print("Vector store ready!")