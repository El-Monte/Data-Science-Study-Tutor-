import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"

def create_full_rag_chain():
    """Creates the conversational RAG chain, cached by Streamlit."""
    with st.spinner("Initializing knowledge base..."):
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 10})
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.1, convert_system_message_to_human=True)

        # This chain rephrases the question based on history
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, "
            "formulate a standalone question which can be understood without the chat history. "
            "Do NOT answer the question. "
            "**If the question is about comparing two or more items, reformulate it into a very simple question "
            "that includes the names of all the items, for example: 'LangChain and CrewAI comparison'.** "
            "Otherwise, return the question as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # This chain answers the question based on retrieved context
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert tutor for data science. Use the following pieces of retrieved context to answer the user's question. If you don't know the answer, just say that you don't know.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        # This is the final chain that ties everything together
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        return rag_chain

# --- Streamlit UI ---
st.set_page_config(page_title="Data Science Tutor")
st.title("🎓 Data Science Study Tutor")

chain = create_full_rag_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello! I am your Data Science Tutor. How can I help?")]

for message in st.session_state.chat_history:
    role = "assistant" if isinstance(message, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(message.content)

if user_prompt := st.chat_input("What is your question?"):
    st.session_state.chat_history.append(HumanMessage(content=user_prompt))
    with st.chat_message("user"):
        st.markdown(user_prompt)
    
    with st.spinner("Thinking..."):
        result = chain.invoke({"input": user_prompt, "chat_history": st.session_state.chat_history})
        response = result["answer"]
    
    st.session_state.chat_history.append(AIMessage(content=response))
    with st.chat_message("assistant"):
        st.markdown(response)