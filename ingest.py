import os
import glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# 1. Configuration
DATA_PATH = "data"

# 2. The Data Ingestion Function
def load_documents():
    """
    Loads all documents from the specified data path. It handles different
    file types (.pdf, .txt, .py, .r) by using the appropriate LangChain loader.

    Returns:
        list[Document]: A list of LangChain Document objects, each containing
                        the content of a file and its metadata (source path).
    """
    all_files = glob.glob(os.path.join(DATA_PATH, "**/*"), recursive=True)
    
    # Filter out directories, keeping only file paths
    files = [f for f in all_files if os.path.isfile(f)]
    
    print(f"Found {len(files)} files in the data directory.")
    
    documents = []
    for file_path in files:
        # Get the file extension to determine which loader to use
        file_extension = os.path.splitext(file_path)[1].lower()
        
        loader = None
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension in ['.txt', '.py', '.md']:
            loader = TextLoader(file_path, encoding='utf-8')
        
        if loader:
            try:
                print(f"Loading: {file_path}")
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                
    return documents

# --- 3. NEW: Document Splitting ---
def split_documents(documents):
    """
    Splits the loaded documents into smaller chunks for better processing.

    Args:
        documents (list[Document]): The list of loaded Document objects.

    Returns:
        list[Document]: A new list of smaller Document chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    
    # Split the documents into chunks
    chunks = text_splitter.split_documents(documents)
    
    print(f"Successfully split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    # Step 1: Ingestion
    print("--- Step 1: Data Ingestion ---")
    documents = load_documents()
    
    if not documents:
        print("No documents were loaded. Exiting.")
    else:
        # Step 2: Splitting
        print("\n--- Step 2: Document Splitting ---")
        chunks = split_documents(documents)
        
        # Let's inspect an example chunk to see the result
        print("\n--- Example of a document chunk ---")
        # Find a chunk that is not the very first one to see overlap
        example_chunk = chunks[1] if len(chunks) > 1 else chunks[0]
        print(f"Content snippet: {example_chunk.page_content}...")
        print(f"Metadata (source): {example_chunk.metadata}")
        print("-----------------------------------")
        
    print("\nIngestion and splitting complete.")