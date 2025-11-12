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
# --- 1. Load Environment Variables ---

load_dotenv()
matplotlib.use("Agg")

# --- 2. Configuration & Constants ---
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"

# --- 3. The RAG Chain (The "Brain" of the App) ---
def create_chains():
    """
       Creates and returns a dictionary containing two chains:
    1. 'rag': The main RAG tutor and plotting chain.
    2. 'explainer': A specialist chain for explaining code.
    """
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.1)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    template = """You are a proactive, expert Data Science tutor and Python programmer. Your main goal is to empower the user by providing clear explanations and fully functional, copy-pastable code examples.

    **Your Persona & Behavior:**
    - Act as a helpful and patient teacher.
    - If a user asks for a concept, first provide a clear definition and explanation. Then, proactively provide a simple, runnable Python code example to demonstrate that concept.
    - Your entire response MUST be a single, valid JSON object with three keys: "explanation", "display_code", and "plot_code".

    **Instructions for JSON Fields:**
    1.  **"explanation"**: This field must contain your full, tutor-like textual explanation.
    2.  **"display_code"**: If you are providing a general Python example (like a function or a class), put the code here. This code is meant to be copied by the user.
    3.  **"plot_code"**: If the user specifically asks for a plot, graph, or visualization, put the Matplotlib code here. This code will be executed.
    4.  **CRITICAL**: You can only populate ONE of the code fields per response. If you provide plot code, `display_code` must be an empty string. If you provide display code, `plot_code` must be an empty string. If no code is relevant, both must be empty strings.

    **Example Interaction:**
    User: "What is Euclidean distance?"
    Assistant (Your Output):
    {{
      "explanation": "Euclidean distance is the straight-line distance between two points. It's calculated using the Pythagorean theorem...",
      "display_code": "import numpy as np\\n\\ndef euclidean_distance(p1, p2):\\n    return np.linalg.norm(np.array(p1) - np.array(p2))\\n\\n# Example usage\\npoint1 = [3, 6]\\npoint2 = [2, 5]\\nprint(f'The distance is: {{euclidean_distance(point1, point2)}}')",
      "plot_code": ""
    }}

    **Now, answer the user's request based on the context and history.**

    PREVIOUS CONVERSATION:
    {chat_history}
    CONTEXT:
    {context}
    QUESTION:
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
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

# --- 4. The Streamlit User Interface ---
st.set_page_config(page_title="Data Science Tutor", layout="wide")
st.title("🎓 Data Science Study Tutor")
st.markdown("Ask a question, ask for a plot, or paste a block of Python code to have it explained!")

# Initialize chains and chat history
if "chains" not in st.session_state:
    with st.spinner("Initializing knowledge base..."):
        st.session_state.chains = create_chains()
    st.success("Knowledge base ready!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Helper function to find and parse JSON ---
def find_and_parse_json(text):
    # This regex finds a JSON block, even with surrounding text
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not json_match:
        # Fallback for when the LLM just returns the JSON without the markdown block
        json_match = re.search(r"(\{.*?\})", text, re.DOTALL)

    if json_match:
        json_str = json_match.group(1)
        return json.loads(json_str)
    # Return None if no JSON is found
    return None

# --- START OF HIGHLIGHTED CHANGES ---
# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Handle complex assistant messages that are now structured dictionaries
        if isinstance(message["content"], dict):
            # Display text explanation if it exists
            if explanation := message["content"].get("explanation"):
                st.markdown(explanation)
            # Display copy-pastable code block if it exists
            if display_code := message["content"].get("display_code"):
                st.code(display_code, language="python") # This adds the copy button
            # Display plot if it exists
            if plot_fig := message["content"].get("plot_fig"):
                st.pyplot(plot_fig)
            # Handle the code explainer's specific structure
            elif message["content"].get("type") == "code_explanation":
                st.markdown(message["content"]["explanation"])
                with st.expander("Show Code and Output"):
                    st.code(message["content"]["code_block"], language="python")
                    st.text("Output:")
                    st.code(message["content"]["code_output"], language="text")
        # Handle simple user messages (which are strings)
        else:
            st.markdown(message["content"])
# --- END OF HIGHLIGHTED CHANGES ---

# React to user input
if user_prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    is_code_block = bool(re.search(r"import |def |for |while |if ", user_prompt.strip()))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # --- IF IT IS A CODE BLOCK ---
            if is_code_block:
                # This 'if' block remains exactly the same as your code.
                output_capture = io.StringIO()
                try:
                    with redirect_stdout(output_capture):
                        exec(user_prompt)
                    code_output = output_capture.getvalue()
                except Exception as e:
                    code_output = f"An error occurred during execution: {e}"
                
                response_str = st.session_state.chains["explainer"].invoke({
                    "code_block": user_prompt,
                    "code_output": code_output
                })

                response_data = find_and_parse_json(response_str)
                if response_data:
                    explanation = response_data.get("explanation", "I couldn't generate an explanation.")
                    st.markdown(explanation)
                    with st.expander("Show Code and Output"):
                        st.code(user_prompt, language="python")
                        st.text("Output:")
                        st.code(code_output, language="text")
                    st.session_state.messages.append({"role": "assistant", "content": {"type": "code_explanation", "explanation": explanation, "code_block": user_prompt, "code_output": code_output}})
                else:
                    st.warning("The explainer model returned a malformed response.")
                    st.markdown(response_str)
                    st.session_state.messages.append({"role": "assistant", "content": response_str})
            
            # --- ELSE, USE THE NORMAL RAG CHAIN ---
            else:
                # --- START OF HIGHLIGHTED CHANGES ---
                history_string = ""
                for message in st.session_state.messages[-5:-1]:
                    # We need to handle the new structured messages in history
                    if isinstance(message["content"], str):
                        history_string += f"{message['role'].capitalize()}: {message['content']}\n"
                    elif isinstance(message["content"], dict) and message["content"].get("explanation"):
                        history_string += f"{message['role'].capitalize()}: {message['content']['explanation']}\n"
                
                response_str = st.session_state.chains["rag"].invoke({
                    "question": user_prompt,
                    "chat_history": history_string
                })
                
                response_data = find_and_parse_json(response_str)
                
                # This is the new logic to handle the 3-field JSON
                if response_data:
                    text_part = response_data.get("explanation", "")
                    display_code_part = response_data.get("display_code", "")
                    plot_code_part = response_data.get("plot_code", "")
                    
                    # Create a single structured message to save in history
                    assistant_message = {
                        "explanation": text_part,
                        "display_code": display_code_part,
                        "plot_fig": None 
                    }

                    if text_part:
                        st.markdown(text_part)
                    
                    if display_code_part:
                        st.code(display_code_part, language="python")
                    
                    if plot_code_part:
                        try:
                            exec_globals = {}
                            exec(plot_code_part, exec_globals)
                            fig = exec_globals.get("fig")

                            if fig:
                                st.pyplot(fig)
                                assistant_message["plot_fig"] = fig # Add the generated figure
                        except Exception as e:
                            st.error(f"An error occurred while executing the plot code: {e}")
                    
                    st.session_state.messages.append({"role": "assistant", "content": assistant_message})

                else: # Fallback for malformed JSON
                    st.warning("The model's response was not in the expected JSON format.")
                    st.markdown(response_str)
                    st.session_state.messages.append({"role": "assistant", "content": response_str})
                # --- END OF HIGHLIGHTED CHANGES ---