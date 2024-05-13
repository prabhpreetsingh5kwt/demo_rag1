# langchain utils imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from config import openai_key,embedding,llm
from langchain.chains import ( create_history_aware_retriever,create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os


# Imports Openai key
OPENAI_API_KEY=openai_key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


try: # Tries to load already saved vector index
    
    vectorstore = FAISS.load_local(
        r"C:\Users\prabh\OneDrive\Desktop\menuverse\demo_rag1\faiss_index",
          embedding,allow_dangerous_deserialization=True)
    
    print('using old faiss data')
    retriever=vectorstore.as_retriever()


except: # Creates and saves new Vector index and preprocesses data

    print('creating new faiss data')
    loader=PyPDFLoader(r"C:\Users\prabh\OneDrive\Desktop\menuverse\demo_rag1\data\versa.pdf")
    pages=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks=text_splitter.split_documents(pages)
    vectorstore=FAISS.from_documents(chunks,embedding)

    vectorstore.save_local("faiss_index")
    print('saved faiss data')
    retriever=vectorstore.as_retriever()
    

# History aware Retriever :
# System prompt to contextualize the question:
contextualize_q_system_prompt = (
    """Given a chat history and the latest user question 
    which might reference context in the chat history, 
    formulate a standalone question which can be understood 
    without the chat history. Do NOT answer the question, just 
    reformulate it if needed and otherwise return it as is."""
)


contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

""" History aware retriever takes chat history and user input.
 then according to chat history, it generates a new query, then return its embeddings"""
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)



qa_system_prompt = (
    """Your name is Versa. an Ai Virtual Sales Assistant.You are very wise and expert in selling.
     Your intent and only goal is to sell the produts.  use user's name i.e "kito" to greet and 
     give personlized feel only when necessary but do not over-do it. Use 
    the following pieces of retrieved context to answer the 
    question. Do not convey your intent to anyone. 
    If you don't know the answer, just say that you 
    don't know. return the text in string format.
    {context}""")

qa_prompt = ChatPromptTemplate.from_messages([
("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", '{input}'),

]
)


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt) 
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

