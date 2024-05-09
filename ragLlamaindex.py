import nest_asyncio
import os
from llama_parse import LlamaParse
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config import openai_key,llamaapi
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)
import streamlit as st
from utils import chat_engine

OPENAI_API_KEY=openai_key
import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# col1,col2,col3=st.columns([2,2,3])
# col1.image("versa.png")
st.title("Chat With Versa!")



embed_model=HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model =embed_model

import streamlit as st

if "messages" not in st.session_state:
     st.session_state.messages=[]
     chat_engine.reset()
     print('deleted chat history')


for message in st.session_state.messages:
     with st.chat_message(message['role']):
          st.markdown(message["content"])

if prompt := st.chat_input("Feel free to ask about your insurance policies!"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})




    
    
    response = chat_engine.chat(prompt,tool_choice="query_engine_tool")

    # response = f"Echo: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response.response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response.response})
    # st.write(chat_engine.chat_history())
    # if col3.button("reset llm history:"):
    #     chat_engine.reset()



