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

load_dotenv()
matplotlib.use("Agg")

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

    rag_template = """
    Your primary function is to act as a JSON API. You MUST respond with a single, valid JSON object and nothing else.
    The JSON object must have two keys: "explanation" and "code".

    You are an expert Data Science tutor. Your goal is to provide a comprehensive, detailed, and insightful answer to the user's QUESTION.
    **CRITICAL LANGUAGE RULE: Your entire response, including all text in the 'explanation', MUST be in the same language as the user's most recent QUESTION. Disregard the language of the previous CHAT HISTORY when deciding the language for your answer.**
    
    **Instructions for JSON content:**
    1.  First and foremost, use the provided CONTEXT as the foundation and primary source of truth for your answer. If the context is in a different language than the question, you must translate the concepts to answer in the user's language.
    2.  After using the context, enrich and expand upon this information with your own broader knowledge to provide a more complete, in-depth explanation.
    3.  The "explanation" value must be a clear, expert-level textual answer to the QUESTION.
    4.  If the QUESTION explicitly asks for a "plot", "graph", "chart", "visualization", or "diagram", you MUST generate complete, runnable Python code to create that visualization in the "code" value. The code must use Matplotlib and create a figure object named 'fig'.
    5.  If the QUESTION asks for a non-plotting code example (like a function or a script), you MUST generate that code in the "code" value.
    6.  If the QUESTION is purely conceptual and does not imply a need for any code, the "code" value MUST be an empty string ("").
    7.  The "explanation" should be self-contained. Do NOT refer to the code (e.g., do not say "the code below...").

    CONTEXT:
    {context}

    CHAT HISTORY:
    {chat_history}

    QUESTION:
    {question}

    **FINAL CHECK: Before you generate the JSON, double-check that the 'explanation' is written in the same language as the QUESTION above.**
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

# Streamlit user interface 
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



# ==========================
# 📚 PRACTICE MODE SECTION
# ==========================

def get_practice_models():
    """
    Build (once per session) a retriever + LLM for Practice Mode.
    We reuse the same constants as the main app (DB_FAISS_PATH, EMBEDDING_MODEL, LLM_MODEL),
    but keep this logic completely separate from the main chat chains.
    """
    if "practice_retriever" not in st.session_state:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
        db = FAISS.load_local(
            DB_FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        st.session_state.practice_retriever = db.as_retriever(search_kwargs={"k": 5})

    if "practice_llm" not in st.session_state:
        st.session_state.practice_llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0.2,
            convert_system_message_to_human=True,
        )

    return st.session_state.practice_retriever, st.session_state.practice_llm


def generate_practice_questions(topic, retriever, llm, n_questions=5):
    """
    Generate N practice questions on a given topic, using course material as context.
    Questions should go from easier (definitions) to harder (reasoning/applications).
    """
    docs = retriever.get_relevant_documents(topic)
    context = "\n\n".join(d.page_content for d in docs[:5])

    prompt = f"""
You are an expert and friendly Data Science tutor.

A student wants to practice the following topic:
"{topic}"

You are given some course MATERIAL (CONTEXT).  
You MUST use this material to understand which concepts to include in the questions,  
BUT the questions themselves MUST be general, universal, and self-contained.

STRICT RULES FOR QUESTIONS:
- Do NOT reference “the text”, “the document”, “the material”, “the context”, “the notes”, or any section/chapter/page.
- Do NOT say “according to the text”, “from the notes”, or anything similar.
- Each question must be phrased as a standard university exam/quiz question.
- Questions must be general (e.g., “What is linear regression?” — not “What does the text say linear regression is?”)
- Start with easy questions (definitions), then medium (reasoning), then hard (applications).

CONTEXT:
\"\"\"{context}\"\"\"

Generate EXACTLY {n_questions} questions.
Output MUST be a numbered list:
1. ...
2. ...
3. ...

Now generate the questions.
"""

    response = llm.invoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)

    questions = []
    for line in text.split("\n"):
        line = line.strip()
        # very simple parsing for lines like "1. ..." "2. ..."
        if line and line[0].isdigit() and "." in line[:4]:
            q = line.split(".", 1)[1].strip()
            questions.append(q)

    if not questions:
        questions = [text.strip()]

    return questions[:n_questions]


def evaluate_practice_answer(question, student_answer, topic, retriever, llm):
    """
    Ask the LLM to evaluate the student's answer using the course context.
    The feedback starts with 'Score: X/100' and then a short explanation.
    """
    docs = retriever.get_relevant_documents(topic + " " + question)
    context = "\n\n".join(d.page_content for d in docs[:5])

    prompt = f"""
You are a Data Science teaching assistant.

You are given:
- some course CONTEXT,
- a QUESTION,
- a STUDENT_ANSWER.

Your evaluation MUST follow these rules:

STRICT RULES:
- DO NOT reference the context or any document.
- DO NOT say “according to the material”, “according to the notes”, “according to the document”.
- Evaluate as if you are a professor who *knows the subject*, not someone reading a text.
- Use ONLY the concepts implied by the context, but NEVER mention that the context exists.

Your output:
- First line MUST be: "Score: X/100" (with X between 0 and 100).
- Then write from 5 to 7 short lines explaining:
  - what is correct,
  - what is missing or inaccurate,
  - the correct explanation the student should have given,
  - how to improve.
- Tone: friendly but rigorous.

CONTEXT:
\"\"\"{context}\"\"\"

QUESTION:
{question}

STUDENT_ANSWER:
{student_answer}
"""

    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


# ---------------------------
# 🎓 Practice Mode UI (Streamlit)
# ---------------------------

st.divider()
st.subheader("📚 Practice Mode (self-assessment)")

# Initialize practice-related state
if "practice_topic" not in st.session_state:
    st.session_state.practice_topic = ""
if "practice_questions" not in st.session_state:
    st.session_state.practice_questions = []
if "practice_index" not in st.session_state:
    st.session_state.practice_index = 0
if "practice_feedback" not in st.session_state:
    st.session_state.practice_feedback = ""
if "practice_answer" not in st.session_state:
    st.session_state.practice_answer = ""

# Topic input
st.session_state.practice_topic = st.text_input(
    "Choose a topic you want to practice (e.g. 'linear regression', 'neural networks', 'variance'):",
    value=st.session_state.practice_topic,
)

col_gen, col_reset = st.columns([2, 1])

with col_gen:
    if st.button("Generate practice questions"):
        if not st.session_state.practice_topic.strip():
            st.warning("Please enter a topic before generating questions.")
        else:
            retriever, llm = get_practice_models()
            st.session_state.practice_questions = generate_practice_questions(
                st.session_state.practice_topic,
                retriever,
                llm,
                n_questions=5,
            )
            st.session_state.practice_index = 0
            st.session_state.practice_feedback = ""
            st.session_state.practice_answer = ""
            st.success(f"Generated {len(st.session_state.practice_questions)} questions on: {st.session_state.practice_topic}")

with col_reset:
    if st.button("Reset Practice Mode"):
        st.session_state.practice_topic = ""
        st.session_state.practice_questions = []
        st.session_state.practice_index = 0
        st.session_state.practice_feedback = ""
        st.session_state.practice_answer = ""
        st.info("Practice Mode has been reset.")

# If we have questions, show the current one
if st.session_state.practice_questions:
    idx = st.session_state.practice_index
    idx_display = idx + 1
    total = len(st.session_state.practice_questions)
    current_question = st.session_state.practice_questions[idx]

    st.markdown(f"**Question {idx_display} / {total}:** {current_question}")

    st.session_state.practice_answer = st.text_area(
        "Your answer:",
        value=st.session_state.practice_answer,
        key="practice_answer_area",
        height=120,
    )

    col_fb, col_next = st.columns([2, 1])

    with col_fb:
        if st.button("Get feedback on this answer"):
            if not st.session_state.practice_answer.strip():
                st.warning("Please write an answer before asking for feedback.")
            else:
                retriever, llm = get_practice_models()
                feedback = evaluate_practice_answer(
                    current_question,
                    st.session_state.practice_answer,
                    st.session_state.practice_topic,
                    retriever,
                    llm,
                )
                st.session_state.practice_feedback = feedback

    with col_next:
        if st.button("Next question"):
            if st.session_state.practice_index < total - 1:
                st.session_state.practice_index += 1
                st.session_state.practice_feedback = ""
                st.session_state.practice_answer = ""
            else:
                st.info("You have reached the last question.")

    if st.session_state.practice_feedback:
        st.markdown("### 🧠 Tutor feedback")
        st.markdown(st.session_state.practice_feedback)
