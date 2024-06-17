# Responsible for Creating/loading Faiss index and Vector store
import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from unstructured.partition.pdf import partition_pdf
from langchain_openai import OpenAIEmbeddings 
from langchain.retrievers import MultiQueryRetriever
from dotenv import load_dotenv
load_dotenv()


openai_key=os.getenv("openai_key")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large",api_key=openai_key)

folder="data"

@st.cache_resource
def create_index():
    try:
        vectorstore = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3})
        print('using old index')
    except:
        print('creating new faiss data')
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
        

        file_path = os.path.join("data/database.txt")
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()


        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        chunks=text_splitter.split_text(text)
        vectorstore=FAISS.from_texts(chunks,embeddings)

        vectorstore.save_local("faiss_index")
        print('saved faiss data')
    return vectorstore
