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
# You are a log analysis expert. 
# You are only allowed to use the following logs to answer questions:

# {logs}

# When answering the user question below:
# 1. Extract relevant information only from the logs.
# 2. Summarize clearly and concisely.
# 3. Highlight ERRORs, WARNINGs, or important events if relevant.
# 4. Provide timestamps if available.

# User question: {question}

# Use bullet points and markdown formatting for readability. Highlight critical issues in **bold**. Include specific timestamps, error codes, and metrics when relevant.
# Keep the output concise and focused on answering the specific question asked.
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
