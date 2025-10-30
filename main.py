import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from vector import retrieve_logs
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# Prompt template
template = """
You are a strict log analysis expert. Use only the logs below to answer questions. 
Do not invent information or refer to anything outside these logs.

{logs}

Rules for answering:
1. Only provide information present in the logs.
2. Ignore irrelevant lines; focus on what’s needed for the user’s question.
3. Highlight **ERRORs, WARNINGs, and critical events** only if relevant.
4. Include timestamps, error codes, or metrics only if they answer the question.
5. Provide concise, structured, actionable answers.
6. If the logs do not contain relevant information, say: 
   "No relevant information found in the logs for this query."
7. Always answer in bullet points or numbered list if multiple points exist.
8. Never list all log lines; only what is useful.

User question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Streamlit UI
st.set_page_config(page_title="Log Analysis Chatbot", layout="wide")
st.title("📊 Log Analysis Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_input := st.chat_input("Type your log question here..."):
    # Display user message
    st.chat_message("user").markdown(user_input)
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Process the question with logs
    with st.spinner("Analyzing logs..."):
        try:
            logs = retrieve_logs(user_input)
            result = chain.invoke({
                "logs": logs,
                "question": user_input
            })

            if hasattr(result, "content"):
                bot_response = result.content.strip()
            elif isinstance(result, dict) and "messages" in result:
                last_msg = result["messages"][-1]
                bot_response = getattr(last_msg, "text", getattr(last_msg, "content", str(last_msg)))
            else:
                bot_response = str(result)

        except Exception as e:
            bot_response = f"Error: {str(e)}"

    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(bot_response)
    # Add bot response to session state
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
