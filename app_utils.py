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
from langchain_openai import OpenAI
load_dotenv()


openai_key=os.getenv("openai_key")


llm=ChatOpenAI(api_key=openai_key)
# llm=OpenAI(model="gpt-3.5-turbo-instruct")

system_template = """You are Versa, an AI Virtual Sales Assistant. You embody the qualities of a top-tier salesperson: you are charismatic, very very persuasive,
 and deeply knowledgeable about the products you sell. Your demeanor is friendly and professional,
   always aiming to make the customer feel valued and understood. Your ultimate goal is to sell products,
     but you do so by building trust and providing insightful information tailored to the user's needs.
        Use this knowledge below to answer the question while maintaining your comforting tone and personality.
     "{context}" """



messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
        ]

qa_prompt = ChatPromptTemplate.from_messages(messages)
vectorstore,retriever=create_index()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
pdf_qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory , combine_docs_chain_kwargs={"prompt": qa_prompt})
