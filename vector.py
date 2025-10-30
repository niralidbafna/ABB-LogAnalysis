from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pathlib import Path
import os

log_folder = "test files"
db_location = "./chroma_langchain_db"

# Create embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Initialize vector store (auto-persist handled by Chroma)
vector_store = Chroma(
    collection_name="logs",
    embedding_function=embeddings,
    persist_directory=db_location
)

def update_vector_store():
    """
    Add any new log files from the folder to the vector store.
    Prints which files were added.
    """
    # Get existing IDs
    existing_ids = set(vector_store._collection.get()["ids"])  # internal collection IDs

    # Find all .log and .txt files
    log_files = list(Path(log_folder).glob("*.log")) + list(Path(log_folder).glob("*.txt"))
    new_documents = []
    new_ids = []

    for file_path in log_files:
        doc_id = str(file_path.name)  # use filename as unique ID
        if doc_id not in existing_ids:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            doc = Document(
                page_content=content,
                metadata={"filename": file_path.name},
                id=doc_id
            )
            new_documents.append(doc)
            new_ids.append(doc_id)

    if new_documents:
        vector_store.add_documents(documents=new_documents, ids=new_ids)
        print("\nNew log files added to vector store:")
        for f in new_ids:
            print(f" - {f}")
    else:
        print("\nNo new log files to add.")

def retrieve_logs(query):
    """
    Updates vector store with new logs and retrieves relevant logs for a query.
    """
    update_vector_store()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return retriever.invoke(query)
