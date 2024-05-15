# langchain utils imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from config import openai_key,embedding,llm
from langchain.chains import ( create_history_aware_retriever,create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from PyPDF2 import PdfReader, PdfWriter
import os
from config import llm
from unstructured.partition.pdf import partition_pdf
from langchain_community.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import SystemMessagePromptTemplate , HumanMessagePromptTemplate,ChatPromptTemplate


# Imports Openai key
OPENAI_API_KEY=openai_key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


try: # Tries to load already saved vector index
    
    vectorstore = FAISS.load_local(
        "faiss_index/",
          embedding,allow_dangerous_deserialization=True)
    
    print('using old faiss data')
    retriever=vectorstore.as_retriever()


except: # Creates and saves new Vector index and preprocesses data

    print('creating new faiss data')

    # Iterates over pdf files and extracts data and finally appending the data to database.txt 
    folder="data"
    pdf_files = [f for f in os.listdir(folder) if f.endswith(".pdf")]

    with open("database.txt", 'a', encoding='utf-8') as file:
        for pdf_file in pdf_files:
            pdf_path = os.path.join(folder, pdf_file)
            print(pdf_file)
            elements = partition_pdf(pdf_path)
            file.write("\n"+"-"*20+"\n")
            file.write(pdf_file+'\n')
            file.write("-"*20+"\n")

            for el in elements:
                file.write((str(el)+"\n"))
    
    file_path = os.path.join("database.txt")

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()


    text_splitter=RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=200)
    chunks=text_splitter.split_text(text)
    vectorstore=FAISS.from_texts(chunks,embedding)

    vectorstore.save_local("faiss_index")
    print('saved faiss data')
    retriever=vectorstore.as_retriever()
    

# History aware Retriever :
# System prompt to contextualize the question:


# contextualize_q_system_prompt = (
#     """Given a chat history and the latest user question 
#     which might reference context in the chat history, 
#     formulate a standalone question which can be understood 
#     without the chat history. Do NOT answer the question, just 
#     reformulate it if needed and otherwise return it as is."""
# )


# contextualize_q_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", contextualize_q_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
# )

# """ History aware retriever takes chat history and user input.
#  then according to chat history, it generates a new query, then return its embeddings"""
# history_aware_retriever = create_history_aware_retriever(
#     llm, retriever, contextualize_q_prompt
# )



# qa_system_prompt = (
#     """Your name is Versa. an Ai Virtual Sales Assistant.You are very wise and expert in selling.
#      Your intent and only goal is to sell the produts.  use user's name i.e "kito" to greet and 
#      give personlized feel only when necessary but do not over-do it. Use 
#     the following pieces of retrieved context to answer the 
#     question. Do not convey your intent to anyone. 
#     If you don't know the answer, just say that you 
#     don't know. return the text in string format.
#     {context}""")

# qa_prompt = ChatPromptTemplate.from_messages([
# ("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", '{input}'),

# ]
# )


# question_answer_chain = create_stuff_documents_chain(llm, qa_prompt) 
# rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

system_template = """Your name is Versa. an Ai Virtual Sales Assistant.You are very wise and expert in selling.
      Your intent and only goal is to sell the produts.  use user's name i.e "kito" to greet and 
      give personlized feel only when necessary but do not over-do it. Use 
     the following pieces of retrieved context to answer the 
     question. Do not convey your intent to anyone. 
     If you don't know the answer, just say that you 
     don't know. return the text in string format.
     {context}"""

messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
        ]
qa_prompt = ChatPromptTemplate.from_messages(messages)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
pdf_qa = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory , combine_docs_chain_kwargs={"prompt": qa_prompt})