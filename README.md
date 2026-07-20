# AI Document Chatbot (RAG-Based)

-> Overview
PDF documents pe sawaal poochho — AI accurate answers dega RAG pipeline use karke banaya hai.
How it Works
1. PDF upload karo
2. Sawaal poochho
3. AI relevant content dhundh ke answer deta hai
Tech Stack
- Python
- LangChain
- Groq LLM (LLaMA 3.1)
- HuggingFace Embeddings
- FAISS Vector Store
- Streamlit
 RAG Pipeline
1. PDF load — PyPDFLoader
2. Chunks — RecursiveCharacterTextSplitter
3. Embeddings — HuggingFace (all-MiniLM-L6-v2)
4. Vector Store — FAISS
5. Retriever — relevant chunks dhundho
6. LLM — answer generate karo

                          ## Projects
                   - Upload any PDF
             - Ask questions about the content
             - Get AI-powered accurate answers

