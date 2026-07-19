import streamlit as st 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq 
st.title("AI Document Chatbot")
st.write("PDF upload karo aur sawaal poochho!") 
uploaded_file = st.file_uploader("upload here", type="pdf")
if uploaded_file is not None:
    st.write("pdf uploaded")
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()
    st.write(f"PDF load ho gaya - {len(pages)} pages!")
    # chunk 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(pages)
    st.write(f"Chunks bane: {len(chunks)}")
    #embedding 
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    st.write("Vector Store ready!") 
    #question 
    question = st.text_input("Sawaal poochho")
    #vector
    if question:
        retriever = vectorstore.as_retriever()
        docs = retriever.invoke(question)
        context = docs[0].page_content
        
        llm = ChatGroq(model="llama-3.1-8b-instant", api_key="")
        
        prompt = f"Context: {context}\n\nQuestion: {question}"
        response = llm.invoke(prompt)
        st.write(response.content)
