# #/*from groq import Groq

# client = Groq(api_key="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")

# response = client.chat.completions.create(
#    model="llama-3.1-8b-instant",
#    messages=[
#     {"role": "system", "content": "You speak in hindi."},
#     {"role": "system", "content": "You are a Data Science teacher. Explain everything simply with examples."},
#     {"role": "user", "content": "What is AI?"}
# ]
# )
#key = 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 
# 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
# print(response.choices[0].message.content) 

from langchain_groq import ChatGroq
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key="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"
)
# response= llm.invoke("what is AI ?")
# print(response.content)
# from langchain_core.prompts import ChatPromptTemplate
# prompt= ChatPromptTemplate.from_messages([
#     ("system","You are a helpful assistant. "),
#     ("user","{question}")
# ]) 
# chain= prompt|llm
# response = chain.invoke({"question":"what is machine learning?"})
# print(response.content) 
#memory 
# from langchain.memory import ConversationBufferMemory 
# memory=ConversationBufferMemory()
# from langchain.Chains import conversationchian
# conversation = conversationchian(
#     llm =llm  ,
#     memory=memory 
# )
# response1 = conversation.predict(input="mera naam sharad hai ")
# print(response1)
# response2 = conversation.predict(input= " mera naam kya hai?") 

# from langchain_core.messages import HumanMessage, AIMessage

# chat_history = []

# chat_history.append(HumanMessage(content="Mera naam Sharad hai"))
# response = llm.invoke(chat_history)
# chat_history.append(AIMessage(content=response.content))
# print(response.content)

# chat_history.append(HumanMessage(content="Mera naam kya hai?"))
# response = llm.invoke(chat_history)
# print(response.content)
#document loader 
from langchain_community.document_loaders import PyPDFLoader 
loader = PyPDFLoader("Sharad_Sourabh_jha_Resume_Final_v6.pdf")
pages= loader.load()
print(len(pages)) 
print(pages[0].page_content) 
#text sp;iter
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
) 
chunks = text_splitter.split_documents(pages)
print(len(chunks)) 
#embedding 
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 
#vector store 
from langchain_community.vectorstores import FAISS
vectorspace = FAISS.from_documents(chunks,embeddings) 
#rag chain
# retriever = vectorspace.as_retriever()
# from langchain_community.chains import RetrievalQA

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever
# )
# result = qa_chain.invoke("What are my skills?")
# print(result['result'])
retriever = vectorspace.as_retriever()
docs = retriever.invoke("What are my skills?")
print(docs[0].page_content)
#proper llm 
context = docs[0].page_content
prompt = f"Context: {context}\n\nQuestion: What are my skills?"
response = llm.invoke(prompt)
print(response.content)