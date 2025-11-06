import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI # <-- CHANGED IMPORT
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# --- 1. Load Environment Variables ---
# Load the OpenAI API key from the .env file
load_dotenv()

# --- 2. Configuration & Constants ---
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"

# --- 3. The RAG Chain (The "Brain" of the App) ---
def create_rag_chain():
    """
    Creates and returns the RAG (Retrieval-Augmented Generation) chain.
    This chain connects all the components: retriever, prompt, model, and parser.
    """
    # Load the vector store and embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    # Create a retriever from the vector store
    # This retriever will find the most relevant documents for a given query.
    retriever = db.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 most relevant chunks

    # Define the prompt template
    # This guides the LLM on how to use the retrieved context to answer the question.
    template = """
    You are an expert tutor specializing in data science.
    Your goal is to provide a clear, concise, and helpful answer to the user's question,
    using only the context provided below. If the context does not contain the answer,
    state that you cannot answer based on the provided information.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Define the LLM
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.1)
    # Helper function to format the retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Create the RAG chain using LangChain Expression Language (LCEL)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- 4. The Streamlit User Interface ---
st.set_page_config(page_title="Data Science Tutor", layout="wide")
st.title("🎓 Data Science Study Tutor")
st.markdown("Ask me anything about statistics, machine learning, Python, R, and more!")

# Initialize the RAG chain and store it in the session state
if "rag_chain" not in st.session_state:
    with st.spinner("Initializing knowledge base... This may take a moment."):
        st.session_state.rag_chain = create_rag_chain()
    st.success("Knowledge base ready!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your question?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get the AI's response using the RAG chain
    with st.spinner("Thinking..."):
        response = st.session_state.rag_chain.invoke(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})