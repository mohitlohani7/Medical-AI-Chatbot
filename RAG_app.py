import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

def load_documents(uploaded_files):
    docs = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            pdf = PdfReader(uploaded_file)
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            docs.append(text)
        elif uploaded_file.type == "text/plain":
            text = uploaded_file.getvalue().decode("utf-8")
            docs.append(text)
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
    return docs

def main():
    st.title("RAG Chatbot")

    uploaded_files = st.file_uploader("Upload PDFs or Text files", type=["pdf", "txt"], accept_multiple_files=True)
    
    if uploaded_files:
        documents = load_documents(uploaded_files)

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = []
        for doc in documents:
            texts.extend(text_splitter.split_text(doc))

        # Create embeddings and vectorstore
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts, embeddings)

        # Create retriever and QA chain
        retriever = vectorstore.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)

        query = st.text_input("Ask your question about the documents:")
        if query:
            with st.spinner("Generating answer..."):
                answer = qa.run(query)
            st.write("**Answer:**", answer)
    else:
        st.info("Please upload one or more PDF or TXT files to start.")

if __name__ == "__main__":
    main()
