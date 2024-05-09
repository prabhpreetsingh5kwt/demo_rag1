# langchain utils
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from config import openai_key,embedding,llm
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import ( create_history_aware_retriever,create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
# from ragLangchain import store

OPENAI_API_KEY=openai_key
import os
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
    
# History aware Retriever

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

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know."
    "{context}")

qa_prompt = ChatPromptTemplate.from_messages([
("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", '{input}'),

]

)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt) 
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# store = {}  # Initialize it to an empty dictionary
# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     global store
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()

#     return store[session_id]

# conversational_rag_chain = RunnableWithMessageHistory(
#     rag_chain,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="chat_history",
#     output_messages_key="answer",
# )