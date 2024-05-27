import os
import openai
import pandas as pd
import base64
import time
import streamlit as st 
from dotenv import load_dotenv
from app_utils import pdf_qa
from langchain_openai import OpenAIEmbeddings
from uptrain import EvalLLM, Evals
from vector_store import create_index
from eval_utils import generate_llm_response
from avatar import get_avatar
load_dotenv()

openai_key=os.getenv("openai_key")
langchain_api_key=os.getenv("langchain_api_key")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ['LANGCHAIN_PROJECT'] = "Versa" 


client = openai.OpenAI()

eval_llm = EvalLLM(openai_api_key=openai_key)

vectorstore,retriever=create_index()

def main():

     st.title("Chat with Versa Using Langchain!")
     # On Every reload:
     # st.session_state.clear()

     if "messages" not in st.session_state:
          st.session_state.clear()

          st.session_state.messages=[]
          # st.session_state.messages.clear()
          link_url = "https://i.ibb.co/D8Z5MHH/versa.png"
          image_html = f"""
    <a href="{link_url}" target="_blank">
        <img src="{link_url}" alt="Image" style='position: fixed;bottom: 113px;
               right: 87px;
               height: 120px;
               width: 120px !important;
               border-radius: 50%;
               z-index: 11;
               '>
    </a>
"""
          st.markdown(image_html, unsafe_allow_html=True)

         

          # Clear conversational buffer memory
          pdf_qa.memory.clear()

          # Initiate Conversation and append it
          st.session_state.messages.append({"role": "assistant", "content": 'Hello Kito, what can i help you with?'})

          


     # Prints chat message history in streamlit ui
     for message in st.session_state.messages:
          with st.chat_message(message['role']):
               st.markdown(message["content"])
    
#      
     # Enters chat prompt and checks if it is not NONE 
     if prompt := st.chat_input("Feel free to ask about your insurance policies!"):
          # Display user message in chat style
          st.chat_message("user").markdown(prompt)
          

          # Add user message to chat history
          st.session_state.messages.append({"role": "user", "content": prompt})


          # Get response from qa chain
          response=pdf_qa({"question": prompt})
          response=str(response['answer']).replace("$","\$")

          #streaming answers
          def char_response_generator(response):
               for char in response:
                    yield char
                    time.sleep(0.001)
          
          # Write response
          with st.chat_message("assistant"):
               st.write_stream(char_response_generator(response))


          # Add assistant response to chat history
          st.session_state.messages.append({"role": "assistant", "content": str(response)})

          
          # Generate evaluation
          questions = [prompt]
          results = []
          for question in questions:
               results.extend(generate_llm_response(question, vectorstore))


          result = eval_llm.evaluate(
          data=results,
          checks=[Evals.CONTEXT_RELEVANCE])
          #Evals.FACTUAL_ACCURACY, Evals.RESPONSE_COMPLETENESS,Evals.RESPONSE_RELEVANCE

          # Try to load evaluation results csv and append the result or create new csv and then append. 
          try:
               
               df = pd.read_csv("evaluation_results.csv")
               print("reading old csv") 
          except FileNotFoundError:
               df = pd.DataFrame()
               print("creating new csv") 


          df = df._append(pd.DataFrame(result),ignore_index=True)


          df.to_csv("evaluation_results.csv", index=False)
          print('evaluation appended')
     

    


if __name__ == "__main__":
    main()