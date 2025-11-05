import os
import glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.docstore.document import Document

# --- 1. Configuration ---
DATA_PATH = "data"

# --- 2. The Data Ingestion Function ---
def load_documents():
    """
    Loads all documents from the specified data path. It handles different
    file types (.pdf, .txt, .py, .r) by using the appropriate LangChain loader.

    Returns:
        list[Document]: A list of LangChain Document objects, each containing
                        the content of a file and its metadata (source path).
    """
    
    # Use glob to find all files in the data directory recursively
    # The pattern '**/*' will match all files in all subdirectories.
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

# --- 3. Main Execution Block ---
if __name__ == "__main__":
    print("Starting data ingestion process...")
    documents = load_documents()
    if documents:
        print(f"\nSuccessfully loaded {len(documents)} documents into memory.")
        print("\n--- Example of a loaded document ---")
        print(f"Content snippet: {documents[0].page_content[:200]}...") # Print the first 200 characters
        print(f"Metadata (source): {documents[0].metadata}")
        print("------------------------------------")
    else:
        print("No documents were loaded. Please check the DATA_PATH and file types.")
        
    print("\nData ingestion complete.")