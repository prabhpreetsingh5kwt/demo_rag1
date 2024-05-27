# Responsible for creating a conversational chain
import os
import random
import string
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate , HumanMessagePromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from vector_store import create_index
from dotenv import load_dotenv

load_dotenv()


openai_key=os.getenv("openai_key")


llm=ChatOpenAI()

system_template = """ Your name is Versa. an Ai Virtual Sales Assistant.You are very wise and expert in selling.
     Your intent and only goal is to sell the produts.  use user's name i.e "kito" to greet and 
      give personlized feel only when necessary but do not over-do it. Use 
     the following pieces of retrieved context to answer the 
     question. Do not convey your intent to anyone. 
     If you don't know the answer, just say that you 
     don't know. return the summarized but good answer.
     "{context}" """



messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
        ]

qa_prompt = ChatPromptTemplate.from_messages(messages)
vectorstore,retriever=create_index()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
pdf_qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory , combine_docs_chain_kwargs={"prompt": qa_prompt})
