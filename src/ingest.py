import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Define the path for the source documents and the vector store
DATA_PATH = '../data'
DB_FAISS_PATH = '../vectorstore/db_faiss'

# --- 1. LOAD a variety of documents ---
def load_documents(data_path):
    """
    Loads documents from the specified data path.
    Supports PDF and Markdown (.md, .txt) files.
    """
    documents = []
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif filename.endswith(('.md', '.txt')):
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
    return documents

# --- 2. SPLIT the documents into smaller chunks ---
def split_documents(documents):
    """
    Splits the documents into smaller chunks for processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    return texts

# --- 3. EMBED the text chunks into numerical vectors ---
def create_embeddings():
    """
    Creates embeddings using a pre-trained model from Hugging Face.
    """
    # We will use a sentence-transformers model.
    # It's powerful, free, and runs locally.
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'} # Use CPU for compatibility
    )
    return embeddings

# --- 4. STORE the embeddings in a FAISS vector store ---
def create_vector_store(texts, embeddings):
    """
    Creates a FAISS vector store from the text chunks and embeddings.
    """
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    return db

# --- Main execution block ---
def main():
    """
    Main function to run the data ingestion process.
    """
    print("Starting data ingestion process...")

    # Load documents from all subdirectories in the data path
    all_documents = []
    for folder in os.listdir(DATA_PATH):
        folder_path = os.path.join(DATA_PATH, folder)
        if os.path.isdir(folder_path):
            print(f"Loading documents from: {folder_path}")
            all_documents.extend(load_documents(folder_path))

    if not all_documents:
        print("No documents found. Please check your data directory.")
        return

    print(f"Loaded {len(all_documents)} documents.")

    # Split the documents into chunks
    texts = split_documents(all_documents)
    print(f"Split documents into {len(texts)} chunks.")

    # Create embeddings
    embeddings = create_embeddings()
    print("Embeddings model loaded.")

    # Create and save the vector store
    print("Creating and saving the vector store...")
    create_vector_store(texts, embeddings)
    print(f"Vector store created and saved at: {DB_FAISS_PATH}")
    print("Ingestion process complete!")

if __name__ == "__main__":
    main()