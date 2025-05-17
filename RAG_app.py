import os
import time
import tkinter as tk
from tkinter import filedialog

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings, CohereEmbeddings, HuggingFaceEmbeddings
from langchain.llms import OpenAI, Cohere, HuggingFaceHub
from langchain.vectorstores import FAISS, Chroma

from loader import load_doc_file  # Assuming you have a loader.py for loading docs


def delete_temp_files():
    """
    Delete temporary files with extensions '.txt' or '.csv' from the current working directory.
    """
    for file in os.listdir(os.getcwd()):
        if file.endswith(".txt") or file.endswith(".csv"):
            try:
                os.remove(file)
                print(f"Deleted temp file: {file}")
            except Exception as e:
                print(f"Failed to delete {file}: {e}")


def select_folder():
    """
    Open a folder selection dialog to allow the user to select a directory.
    Returns the selected directory path.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    folder_selected = filedialog.askdirectory()
    root.destroy()
    return folder_selected


def load_vectorstore(vstore_path: str, vstore_name: str, embedding_model):
    """
    Load a vectorstore either from FAISS or Chroma based on vstore_name.
    Parameters:
        vstore_path (str): Path where the vectorstore is stored.
        vstore_name (str): 'faiss' or 'chroma' specifying the vectorstore type.
        embedding_model: Embedding model instance to use with the vectorstore.
    Returns:
        Loaded vectorstore instance.
    """
    try:
        if vstore_name == "faiss":
            vectorstore = FAISS.load_local(vstore_path, embedding_model)
        elif vstore_name == "chroma":
            vectorstore = Chroma(persist_directory=vstore_path, embedding_function=embedding_model)
        else:
            raise ValueError("Unsupported vectorstore type. Use 'faiss' or 'chroma'.")
        return vectorstore
    except Exception as e:
        st.error(f"Failed to load vectorstore: {e}")
        return None


def create_retriever(
    base_retriever_search_type: str = "similarity",
    base_retriever_search_k: int = 15,
    use_score_threshold: bool = False,
    score_threshold: float = 0.3,
    vectorstore=None,
):
    """
    Create a retriever object with specified search type and parameters.
    Parameters:
        base_retriever_search_type (str): 'similarity' or 'mmr' (Maximal Marginal Relevance).
        base_retriever_search_k (int): Number of top documents to retrieve.
        use_score_threshold (bool): Whether to filter documents by score threshold.
        score_threshold (float): Minimum score threshold for document retrieval.
        vectorstore: The vectorstore instance to create retriever from.
    Returns:
        A retriever object.
    """
    if vectorstore is None:
        raise ValueError("Vectorstore cannot be None when creating retriever.")
    
    if base_retriever_search_type == "similarity":
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": base_retriever_search_k})
    elif base_retriever_search_type == "mmr":
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": base_retriever_search_k})
    else:
        raise ValueError("Invalid search type. Choose 'similarity' or 'mmr'.")
    
    if use_score_threshold:
        retriever.search_kwargs["score_threshold"] = score_threshold

    return retriever


def get_embedding_model(llm_choice: str, hf_api_key: str = None):
    """
    Initialize and return the embedding model instance based on llm_choice.
    Parameters:
        llm_choice (str): One of 'openai', 'cohere', or 'huggingface'.
        hf_api_key (str): HuggingFace API key (optional).
    Returns:
        Embedding model instance.
    """
    if llm_choice == "openai":
        return OpenAIEmbeddings()
    elif llm_choice == "cohere":
        return CohereEmbeddings()
    elif llm_choice == "huggingface":
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    else:
        raise ValueError(f"Unsupported embedding model choice: {llm_choice}")


def get_llm_model(llm_choice: str, openai_api_key: str = "", google_api_key: str = "", cohere_api_key: str = "", hf_api_key: str = ""):
    """
    Initialize and return the LLM instance based on llm_choice.
    """
    if llm_choice == "openai":
        return OpenAI(openai_api_key=openai_api_key, temperature=0)
    elif llm_choice == "cohere":
        return Cohere(cohere_api_key=cohere_api_key, temperature=0)
    elif llm_choice == "huggingface":
        return HuggingFaceHub(hf_api_key=hf_api_key, temperature=0)
    else:
        raise ValueError(f"Unsupported LLM choice: {llm_choice}")


def main():
    st.title("Retrieval-Augmented Generation (RAG) Chatbot")

    # Initialize API keys in session state if not present
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""
    if "google_api_key" not in st.session_state:
        st.session_state.google_api_key = ""
    if "cohere_api_key" not in st.session_state:
        st.session_state.cohere_api_key = ""
    if "hf_api_key" not in st.session_state:
        st.session_state.hf_api_key = ""

    # Sidebar for API keys and model selection
    with st.sidebar:
        st.header("API Keys")
        st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.openai_api_key)
        st.session_state.google_api_key = st.text_input("Google API Key", type="password", value=st.session_state.google_api_key)
        st.session_state.cohere_api_key = st.text_input("Cohere API Key", type="password", value=st.session_state.cohere_api_key)
        st.session_state.hf_api_key = st.text_input("HuggingFace API Key", type="password", value=st.session_state.hf_api_key)

        st.header("Model Selection")
        llm_choice = st.selectbox("Choose LLM Model", ["openai", "cohere", "huggingface"])
        embedding_choice = st.selectbox("Choose Embedding Model", ["openai", "cohere", "huggingface"])
        vectorstore_choice = st.selectbox("Choose Vectorstore", ["faiss", "chroma"])

    # Upload or select documents
    uploaded_files = st.file_uploader("Upload document files (PDF, TXT, CSV)", accept_multiple_files=True)

    docs = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            content = load_doc_file(uploaded_file)
            if content:
                docs.append(content)
        st.success(f"Loaded {len(docs)} documents.")

    # Select or enter vectorstore path
    vstore_path = st.text_input("Vectorstore path", value="")
    if st.button("Browse Vectorstore Folder"):
        folder = select_folder()
        if folder:
            vstore_path = folder
            st.experimental_rerun()

    # Create embeddings
    try:
        embedding_model = get_embedding_model(embedding_choice, st.session_state.hf_api_key)
    except Exception as e:
        st.error(f"Error initializing embedding model: {e}")
        return

    # Load or create vectorstore
    vectorstore = None
    if vstore_path and os.path.exists(vstore_path):
        vectorstore = load_vectorstore(vstore_path, vectorstore_choice, embedding_model)
    else:
        if docs:
            # You would add logic here to create a new vectorstore from docs
            st.info("Vectorstore path invalid or empty. Please provide a valid path or create a vectorstore.")

    # Create retriever
    retriever = None
    if vectorstore:
        try:
            retriever = create_retriever(
                base_retriever_search_type="similarity",
                base_retriever_search_k=15,
                vectorstore=vectorstore,
            )
        except Exception as e:
            st.error(f"Error creating retriever: {e}")

    # Initialize LLM model
    try:
        llm = get_llm_model(
            llm_choice,
            openai_api_key=st.session_state.openai_api_key,
            cohere_api_key=st.session_state.cohere_api_key,
            hf_api_key=st.session_state.hf_api_key,
        )
    except Exception as e:
        st.error(f"Error initializing LLM model: {e}")
        return

    # Setup QA chain if retriever and llm available
    if retriever and llm:
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        query = st.text_input("Enter your question here:")
        if st.button("Ask") and query.strip():
            with st.spinner("Generating answer..."):
                try:
                    answer = qa_chain.run(query)
                    st.markdown(f"**Answer:** {answer}")
                except Exception as e:
                    st.error(f"Failed to get answer: {e}")
    else:
        st.warning("Please load a vectorstore and initialize models to start querying.")

    if st.button("Clear Temporary Files"):
        delete_temp_files()
        st.success("Temporary files deleted.")


if __name__ == "__main__":
    main()
