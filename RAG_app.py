import os

# Disable CUDA to force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def load_llm(repo_id):
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"},
    )

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

def main():
    # Load FAISS Vectorstore
    DB_FAISS_PATH = "vectorstore/db_faiss"
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # Force CPU usage to avoid Torch device errors
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        index_name="index",  # ← match your file name index.faiss and index.pkl
        allow_dangerous_deserialization=True
    )

    # Setup QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(HUGGINGFACE_REPO_ID),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
    )

    # Example query
    query = input("Write your query here: ")
    response = qa_chain.invoke({"query": query})

    print("\nAnswer:\n", response["result"])
    print("\nSource Documents:\n", response["source_documents"])

if __name__ == "__main__":
    main()
