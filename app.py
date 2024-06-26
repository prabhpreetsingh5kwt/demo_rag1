import os
import openai
import pandas as pd
import base64
import time
import streamlit as st 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from app_utils import check_and_create_csv,append_to_csv,generate_unique_filename,pdf_qa,sentiment
import urllib.request
from secondarydb.semantic_srch_utils import search_with_faiss
from agent import preprocess
from avatar import get_avatar
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory

load_dotenv()

openai_key=os.getenv("openai_key")
langchain_api_key=os.getenv("langchain_api_key")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ['LANGCHAIN_PROJECT'] = "Versa" 
llm=ChatOpenAI()
client = openai.OpenAI()

# Create a directory for storing videos if it doesn't exist
os.makedirs('videos', exist_ok=True)


def main():
     session=st.text_input('enter your session name')
     memory = MongoDBChatMessageHistory(connection_string="mongodb+srv://prabhpreets5kwt:w5jtynqcht6sq3r9@cluster0.ma2wfua.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",session_id=session)

    
     st.title("Chat with Versa Using Langchain!")

     # On Every reload empty the messages, add versa's image, clear rag's memory and append an AI message:
     if "messages" not in st.session_state:
          st.session_state.messages=[]
          # st.session_state.messages.clear()
          

          # add versa's image on bottom right 
          link_url = "https://i.ibb.co/8Y6QcVt/versa.png"
          image_html = f"""
    <a href="{link_url}" target="_blank">
        <img src="{link_url}" alt="Image" style='position: fixed;bottom: 113px;
               right: 87px;
               height: 120px;
               width: 120px ;
               border-radius: 50%;
               z-index: 11;
               object-fit:cover;
               '>
    </a>
"""
          st.markdown(image_html, unsafe_allow_html=True)

          
          # Initiate Conversation and append it
          st.session_state.messages.append({"role": "assistant", "content": f'Hello, what can i help you with?'})

     with st.sidebar:
          state=st.text_input('state')
     
     # Prints chat message history in streamlit ui
     for message in st.session_state.messages:
          with st.chat_message(message['role']):
               st.markdown(message["content"])
    

     #trivia agent executor     
     executor=preprocess()

     # Enters chat prompt and checks if it is not NONE 
     if prompt := st.chat_input("Feel free to ask about your insurance policies!"):
          # Display user message in chat style
          st.chat_message("user").markdown(prompt)
          
          # Add user message to chat history
          st.session_state.messages.append({"role": "user", "content": prompt})

          # Get response from qa chain
          start_time = time.time()


          #semantic search:
          response_faiss,video_path,score=search_with_faiss(prompt)

          if score>0.50:
               response=response_faiss
               st.write(score)
               memory.add_user_message(response["question"])

          

          else:



               response=pdf_qa({"question": prompt,"chat_history": memory.messages})
               rephrased_query=response['generated_question']
               response=str(response['answer']).replace("$","\$")
          
          # memory.add_user_message(response["question"])
          # memory.add_ai_message(response["answer"])
               print(rephrased_query)          


          #sentiment analyzer and product extractor for user query which returns a query for trivia agent
               sentiment_response=sentiment(rephrased_query)
               if sentiment_response['sentiment']=='negative':
                    product=sentiment_response['product']
                    query=f"{product} in {state}"
                    print(query)
               
                    #executor block for trivia agent
                    exec_response = executor.invoke({"input": query, "chat_history": []})
                    exec_response=exec_response['output']
                    exec_response=exec_response.replace("$","\$")
                    
                    #Add trivia to response
                    response=response+' '+ '\n\n'+exec_response


          #"""pass response to avatar function and show video  on webpage""" 
          # result_url=get_avatar(response)
#           video_html = f"""
#           <video controls autoplay  style='position: fixed;bottom: 113px;
#            right: 87px;
#            height: 120px;
#            width: 120px;
#            border-radius: 50%;
#            z-index: 11;
#            object-fit: cover;'>
#           <source src="{result_url}" type="video/mp4">
#           Your browser does not support the video tag.
#           </video>"""
# 
#           # Display the HTML in Streamlit
#           st.markdown(video_html, unsafe_allow_html=True)
          end_time = time.time()
          elapsed_time = end_time - start_time

          #streaming answers
          def char_response_generator(response):
               for char in response:
                    yield char
                    time.sleep(0.001)
          

          # Write response
          with st.chat_message("assistant"):
               st.write_stream(char_response_generator(response))
          st.write(f"Time taken: {elapsed_time:.2f} seconds")

          # Add assistant response to chat history
          st.session_state.messages.append({"role": "assistant", "content": str(response)})
          
          # Appending data in data source
          video_path = generate_unique_filename()
          
          #"""Download video and read dataframe"""

          # video=urllib.request.urlretrieve(result_url,video_path)
          # source='secondarydb/datasource.csv'
          # check_and_create_csv(source)

          #"""Append query response and video path to secondary database"""
          # append_to_csv(source,rephrased_query,response,video_path)
          # 
          # NEED TO CREATE EMBEDDINGS AGAIN AND UPDATE THE FAISS INDEX FOR SECONDARYDB         

          

if __name__ == "__main__":
    main()