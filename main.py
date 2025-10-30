from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate  # type: ignore
# from vector import retriever
from vector import retrieve_logs


import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# Prompt template
template = """
You are a log analysis expert. 
You are only allowed to use the following logs to answer questions:

{logs}

When answering the user question below:
1. Extract relevant information only from the logs.
2. Summarize clearly and concisely.
3. Highlight ERRORs, WARNINGs, or important events if relevant.
4. Provide timestamps if available.

User question: {question}

Use bullet points and markdown formatting for readability. Highlight critical issues in **bold**. Include specific timestamps, error codes, and metrics when relevant.
Keep the output concise and focused on answering the specific question asked.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


def print_separator(char="=", length=50):
    print(f"\n{char * length}\n")


def display_result(result):
    """
    Handles and prints the LLM result cleanly.
    """
    if hasattr(result, "content"):
        # Standard Gemini result
        print(result.content.strip())
    elif isinstance(result, dict) and "messages" in result:
        last_msg = result["messages"][-1]
        if hasattr(last_msg, "text"):
            print(last_msg.text.strip())
        elif hasattr(last_msg, "content"):
            print(last_msg.content.strip())
        else:
            print(str(last_msg))
    else:
        print(str(result))


# Main loop
while True:
    print_separator("-")
    print("LOG ANALYSIS SYSTEM")
    print_separator("-")

    question = input("Enter your question about the logs (or 'q' to quit): ").strip()
    if question.lower() == 'q':
        print("\nExiting log analysis system. Goodbye!")
        break

    print_separator()
    print("Analyzing logs...\n")

    try:
        # logs = retriever.invoke(question)
        logs = retrieve_logs(question)
        result = chain.invoke({
            "logs": logs,
            "question": question
        })

        print_separator()
        print("ANALYSIS RESULTS:\n")
        display_result(result)
        print_separator()

    except Exception as e:
        print_separator()
        print("ERROR: An error occurred during analysis:")
        print(f"  {str(e)}")
        print_separator()
