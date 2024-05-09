
import os
import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import ( create_history_aware_retriever,create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from raglang_utils import rag_chain
from config import openai_key,llm


OPENAI_API_KEY=openai_key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


st.title("Chat with Versa Using Langchain!")


if "messages" not in st.session_state:
     st.session_state.messages=[]


if "stores" not in st.session_state:
    st.session_state.stores=[]


for message in st.session_state.messages:
     with st.chat_message(message['role']):
          st.markdown(message["content"])


store = {}  # Initialize it to an empty dictionary

# for prstore in st.session_state.stores:
#     store.append(prstore)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    global store
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)  


if prompt := st.chat_input("Feel free to ask about your insurance policies!"):
# Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = conversational_rag_chain.invoke(
        {"input": prompt},
        config={
            "configurable": {"session_id": "abc123"}
        },  # constructs a key "abc123" in `store`.
    )["answer"]
    print(store)
    
    st.session_state.stores.append(store)

    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})




