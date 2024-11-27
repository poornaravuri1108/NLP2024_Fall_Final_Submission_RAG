from typing import List
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
import shutil
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env', verbose=True)
NVIDIA_E5_EMBEDDING_API_KEY = os.getenv('NVIDIA_E5_EMBEDDING_API_KEY')

def load_documents(DATA_PATH: str) -> List[Document]:
    documents = []
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Directory not found: {DATA_PATH}")
    
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files in {DATA_PATH}")
    
    for filename in pdf_files:
        filepath = os.path.join(DATA_PATH, filename)
        try:
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            
            for doc in docs:
                doc.metadata['source'] = filename
            
            documents.extend(docs)
            print(f"Successfully loaded {len(docs)} pages from {filename}")
            
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
    
    return documents

def split_text(documents: List[Document]) -> List[Document]:
    if not documents:
        raise ValueError("No documents to split")
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"{len(documents)} documents are split into {len(chunks)} chunks")
    
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    print(f"After removing empty chunks: {len(chunks)} chunks remain")
    
    return chunks

def save_data_to_db(data_chunks: List[Document], CHROMA_PATH: str):
    if not data_chunks:
        raise ValueError("No chunks to save to database")
        
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    try:
        # embeddings = NVIDIAEmbeddings(
        #     model="nvidia/nv-embedqa-e5-v5",
        #     api_key=NVIDIA_E5_EMBEDDING_API_KEY,
        #     truncate="NONE"
        # )   
        # embeddings = NVIDIAEmbeddings(
        #     model="nvidia/nv-embedqa-e5-v5", 
        #     api_key="", 
        #     truncate="NONE", 
        # ) 

        embeddings = NVIDIAEmbeddings(
                model="nvidia/nv-embedqa-mistral-7b-v2", 
                api_key="", 
                truncate="NONE", 
                )    
        
        vector_store = Chroma(
            collection_name="NLP_Project_embedding",
            embedding_function=embeddings,
            persist_directory=CHROMA_PATH
        )

        vector_store.add_documents(documents=data_chunks)

        print(f"Saved {len(data_chunks)} chunks to chroma db")
        return vector_store
        
    except Exception as e:
        print(f"Error creating embeddings: {str(e)}")
        raise

def getEmbeddings(CHROMA_PATH: str, DATA_PATH: str):
    try:
        documents = load_documents(DATA_PATH)
        if not documents:
            raise ValueError("No documents were successfully loaded")
            
        data_chunks = split_text(documents)
        if not data_chunks:
            raise ValueError("No chunks were created from the documents")
            
        db = save_data_to_db(data_chunks, CHROMA_PATH)
        return db
        
    except Exception as e:
        print(f"Error in getEmbeddings: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        if not NVIDIA_E5_EMBEDDING_API_KEY:
            raise ValueError("Nvidia embedding api key not found in environment variables")
        else:
            print(f"NVIDIA EMBEDDING API KEY IS {NVIDIA_E5_EMBEDDING_API_KEY}")
            
        CHROMA_PATH = "chroma"
        DATA_PATH = "pdfs/"
        
        os.makedirs(CHROMA_PATH, exist_ok=True)
        
        db = getEmbeddings(CHROMA_PATH, DATA_PATH)
        
        if db is not None:
            test_results = db.similarity_search("test query", k=1)
            print("\nDatabase test successful")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")