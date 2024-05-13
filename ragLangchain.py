# Imports
import os
import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import ( create_history_aware_retriever,create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from raglang_utils import rag_chain
from config import openai_key,llm

# Import openai api key
os.environ["OPENAI_API_KEY"] = openai_key


# Creates a title
st.title("Chat with Versa Using Langchain!")


# Creates a messages list
if "messages" not in st.session_state:
     st.session_state.messages=[]


# Create a stores list to preserve the value of "store variable"
if "stores" not in st.session_state:
    st.session_state.stores=[]


# Prints chat message history in streamlit ui
for message in st.session_state.messages:
     with st.chat_message(message['role']):
          st.markdown(message["content"])
print('st.session_state.stores',st.session_state.stores)


# If stores list is populated then add the value to the "store" variable :
if st.session_state.stores:
    print("st.session_state.stores::::",st.session_state.stores)
    store = st.session_state.stores   # Initialize it to an empty dictionary
else:
    store={}

# Get_session_history function is responsible for returning a chat message history with respect to a particular session_id 
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # if a session_id is in store, it will retrieve its data, else it will create a store with new session id
    global store
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    return store[session_id]

# Defining conversational rag chain which includes rag_chain, and session history
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)  

#if is not none:
if prompt := st.chat_input("Feel free to ask about your insurance policies!"):


    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # response
    response = conversational_rag_chain.invoke(
        {"input": prompt},
        config={
            "configurable": {"session_id": "abc123"}
        },  # constructs a key "abc123" in `store`.
    )["answer"]
    print("populated store------------->",store)
    
    # Populates stores list for session storage
    st.session_state.stores=store

    # human_message_str=st.session_state.stores["abc123"].messages[0].content
    # ai_message_str=st.session_state.stores["abc123"].messages[1].content
   
    # Print assistant message :
    with st.chat_message("assistant"):
        st.write(str(response))

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": str(response)})
