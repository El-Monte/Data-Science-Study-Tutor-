import streamlit as st
import os
import matplotlib
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from operator import itemgetter
import json
import io
from contextlib import redirect_stdout
import re
import seaborn as sns

# --- 1. Load Environment Variables ---
load_dotenv()
matplotlib.use("Agg")


# --- 2. Configuration & Constants ---
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"


def create_chains():
    """
    Creates and returns a dictionary containing two specialized chains:
    1. 'rag': The main RAG tutor for questions and plotting.
    2. 'explainer': A specialist chain for explaining code.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.1, convert_system_message_to_human=True)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # --- Chain 1: The RAG Tutor and Plotter (with a NEW, SIMPLER, MORE RELIABLE PROMPT) ---
    rag_template = """
    Your primary function is to act as a JSON API. You MUST respond with a single, valid JSON object and nothing else.
    The JSON object must have two keys: "explanation" and "code".

    You are an expert Data Science tutor. Use the provided CONTEXT and CHAT HISTORY to answer the user's QUESTION.

    **Instructions for JSON content:**
    1.  The "explanation" value must be a clear, expert-level textual answer to the QUESTION.
    2.  If the QUESTION explicitly asks for a "plot", "graph", "chart", "visualization", or "diagram", you MUST generate complete, runnable Python code to create that visualization in the "code" value. The code must use Matplotlib and create a figure object named 'fig'.
    3.  If the QUESTION asks for a non-plotting code example (like a function or a script), you MUST generate that code in the "code" value.
    4.  If the QUESTION is purely conceptual and does not imply a need for any code, the "code" value MUST be an empty string ("").
    5.  The "explanation" should be self-contained. Do NOT refer to the code (e.g., do not say "the code below...").

    CONTEXT:
    {context}

    CHAT HISTORY:
    {chat_history}

    QUESTION:
    {question}
    """
    rag_prompt = ChatPromptTemplate.from_template(rag_template)
    
    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # --- Chain 2: The Code Explainer ---
    code_explainer_template = """You are an expert Python code explainer.
    The user has provided a piece of code, and I have already run it for you.
    Your task is to explain what the code does, step by step, and present the output.

    CODE:
    ```python
    {code_block}
    ```

    EXECUTION OUTPUT:
    ```
    {code_output}
    ```

    Your response MUST be a JSON object with a single key: "explanation".
    The "explanation" should be a clear, step-by-step breakdown of the code's logic and what the final output means.
    """
    code_explainer_prompt = ChatPromptTemplate.from_template(code_explainer_template)
    code_explainer_chain = code_explainer_prompt | llm | StrOutputParser()

    return {"rag": rag_chain, "explainer": code_explainer_chain}

# --- 4. Helper Function for the UI (FIX: Defined only ONCE) ---
def find_and_parse_json(text: str):
    """Finds and parses the first valid JSON object in a string."""
    try:
        start_index = text.find('{')
        end_index = text.rfind('}') + 1
        if start_index != -1 and end_index != 0:
            json_str = text[start_index:end_index]
            return json.loads(json_str)
    except json.JSONDecodeError:
        return None
    return None

# --- 5. The Streamlit User Interface (with corrected display logic) ---
st.set_page_config(page_title="Data Science Tutor", layout="wide")
st.title("🎓 Data Science Study Tutor")
st.markdown("Ask a question, ask for a plot, or paste a block of Python code to have it explained!")

if "chains" not in st.session_state:
    with st.spinner("Initializing knowledge base..."):
        st.session_state.chains = create_chains()
    st.success("Knowledge base ready!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        if isinstance(content, dict):
            if "explanation" in content and content["explanation"]:
                st.markdown("### 💡 Explanation")
                st.markdown(content["explanation"])
                st.divider()
            if "code" in content and content["code"]:
                st.markdown("### 🐍 Generated Code")
                st.code(content["code"], language="python")
            if "fig" in content:
                st.markdown("### 📊 Generated Plot")
                st.pyplot(content["fig"])
            if "code_block" in content:
                 st.markdown("### 🔬 Code Breakdown")
                 st.markdown(content.get("explanation", ""))
                 with st.expander("Show Executed Code and Output"):
                    st.code(content["code_block"], language="python")
                    st.text("Output:")
                    st.code(content["code_output"], language="text")
        else:
            st.markdown(content)

# React to new user input
if user_prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            is_code_block = bool(re.search(r"^\s*(import|def|for|while|if|#)", user_prompt.strip())) or len(user_prompt.strip().split('\n')) > 1

            # --- ROUTE 1: Handle Code Explanation ---
            if is_code_block:
                code_to_explain = user_prompt
                
                output_capture = io.StringIO()
                try:
                    with redirect_stdout(output_capture):
                        exec(code_to_explain)
                    code_output = output_capture.getvalue()
                except Exception as e:
                    code_output = f"An error occurred during execution: {e}"
                
                response_str = st.session_state.chains["explainer"].invoke({ "code_block": code_to_explain, "code_output": code_output })
                response_data = find_and_parse_json(response_str)

                if response_data and "explanation" in response_data:
                    explanation = response_data["explanation"]
                    st.markdown("### 🔬 Code Breakdown")
                    st.markdown(explanation)
                    with st.expander("Show Executed Code and Output"):
                        st.info("This is the code that was executed:")
                        st.code(code_to_explain, language="python")
                        st.text("Captured Output:")
                        st.code(code_output, language="text")
                    
                    st.session_state.messages.append({ "role": "assistant", "content": { "explanation": explanation, "code_block": code_to_explain, "code_output": code_output } })
                else:
                    st.error("I had trouble explaining that code. Here is the raw response:")
                    st.code(response_str, language="text")
                    st.session_state.messages.append({"role": "assistant", "content": response_str})

            # --- ROUTE 2: Handle RAG Question ---
            else:
                history_string = ""
                for message in st.session_state.messages[-5:-1]:
                    content = message["content"]
                    if isinstance(content, str):
                        history_string += f"{message['role'].capitalize()}: {content}\n"
                    elif isinstance(content, dict) and "explanation" in content:
                        history_string += f"{message['role'].capitalize()}: {content['explanation']}\n"
                
                response_str = st.session_state.chains["rag"].invoke({ "question": user_prompt, "chat_history": history_string })
                response_data = find_and_parse_json(response_str)

                if response_data:
                    explanation = response_data.get("explanation", "")
                    generated_code = response_data.get("code", "")
                    response_content = {}
                    
                    if explanation:
                        st.markdown("### 💡 Explanation")
                        st.markdown(explanation)
                        response_content["explanation"] = explanation
                    
                    if generated_code:
                        st.divider()
                        if "fig" in generated_code or "plt.figure" in generated_code:
                            st.markdown("### 📊 Generated Plot")
                            try:
                                exec_globals = {}
                                exec(generated_code, exec_globals)
                                fig = exec_globals.get("fig")
                                if fig:
                                    st.pyplot(fig)
                                    response_content["fig"] = fig
                            except Exception as e:
                                st.error(f"An error occurred while generating the plot: {e}")
                        else:
                            st.markdown("### 🐍 Generated Code")
                            st.code(generated_code, language="python")
                            response_content["code"] = generated_code
                    
                    if response_content:
                        st.session_state.messages.append({"role": "assistant", "content": response_content})
                else:
                    st.error("I had trouble formatting my response. Here is the raw output:")
                    st.code(response_str, language="text")
                    st.session_state.messages.append({"role": "assistant", "content": response_str})