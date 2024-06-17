import os
import uuid
import openai
import pandas as pd
import base64
import time
import streamlit as st 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from app_utils import pdf_qa
from app_utils import download_video,check_and_create_csv,append_to_csv,generate_unique_filename
from uptrain import EvalLLM, Evals
from vector_store import create_index
from eval_utils import generate_llm_response
from app_utils import sentiment
import urllib.request
from agent import preprocess
from avatar import get_avatar
load_dotenv()

openai_key=os.getenv("openai_key")
langchain_api_key=os.getenv("langchain_api_key")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ['LANGCHAIN_PROJECT'] = "Versa" 

# Create a directory for storing videos if it doesn't exist
os.makedirs('videos', exist_ok=True)


client = openai.OpenAI()

eval_llm = EvalLLM(openai_api_key=openai_key)
llm=ChatOpenAI()
vectorstore,retriever=create_index()

def main():

     st.title("Chat with Versa Using Langchain!")
     # On Every reload:
     # st.session_state.clear()

     

     if "messages" not in st.session_state:
          # st.session_state.clear()

          st.session_state.messages=[]
          # st.session_state.messages.clear()
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

         

          # Clear conversational buffer memory
          pdf_qa.memory.clear()
     
          

          # Initiate Conversation and append it
          st.session_state.messages.append({"role": "assistant", "content": 'Hello kito, what can i help you with?'})

          


     # Prints chat message history in streamlit ui
     for message in st.session_state.messages:
          with st.chat_message(message['role']):
               st.markdown(message["content"])
    
     
     
          
     with st.sidebar:
          state=st.text_input('state')
          # name=st.text_input('name')
          
     

     executor=preprocess()

     # Enters chat prompt and checks if it is not NONE 
     if prompt := st.chat_input("Feel free to ask about your insurance policies!"):
          # Display user message in chat style
          st.chat_message("user").markdown(prompt)
          

          # Add user message to chat history
          st.session_state.messages.append({"role": "user", "content": prompt})


          # Get response from qa chain
          start_time = time.time()
          response=pdf_qa({"question": prompt})
          rephrased_query=response['generated_question']
          response=str(response['answer']).replace("$","\$")
          print(rephrased_query)
          print(type(rephrased_query))
          
#  

          sentiment_response=sentiment(rephrased_query)
          if sentiment_response['sentiment']=='negative':
               product=sentiment_response['product']
               query=f"{product} in {state}"
               print(query)
               exec_response = executor.invoke({"input": query, "chat_history": []})
               exec_response=exec_response['output']
               exec_response=exec_response.replace("$","\$")
               
               response=response+' '+ '\n\n'+exec_response

          #executor block for trivia agent

          
#           result_url=get_avatar(response)


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
#           </video>
# """

# # Display the HTML in Streamlit
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
          # st.write(pdf_qa.memory.chat_memory.messages)
          # st.write(rephrased_ques)
          # print(rephrased_query)
          


          st.write(f"Time taken: {elapsed_time:.2f} seconds")
          # Add assistant response to chat history
          st.session_state.messages.append({"role": "assistant", "content": str(response)})
          
          def generate_unique_filename():
               while True:
                    unique_id = uuid.uuid4()
                    video_name = f'video_{unique_id}.mp4'
                    video_path = os.path.join('videos', video_name)
                    if not os.path.exists(video_path):
                         return video_path
          # Appending data in data source
          video_path = generate_unique_filename()

          # video=urllib.request.urlretrieve(result_url,video_path)
          # source='datasource.csv'
          # check_and_create_csv(source)

          # append_to_csv(source,rephrased_query,response,video_path)          

          # Generate evaluation
          # questions = [prompt]
          # results = []
          # for question in questions:
          #      results.extend(generate_llm_response(question, vectorstore))


          # result = eval_llm.evaluate(
          # data=results,
          # checks=[Evals.CONTEXT_RELEVANCE])
          #Evals.FACTUAL_ACCURACY, Evals.RESPONSE_COMPLETENESS,Evals.RESPONSE_RELEVANCE

          # Try to load evaluation results csv and append the result or create new csv and then append. 
          # try:
               
          #      df = pd.read_csv("evaluation_results.csv")
          #      print("reading old csv") 
          # except FileNotFoundError:
          #      df = pd.DataFrame()
          #      print("creating new csv") 


          # df = df._append(pd.DataFrame(result),ignore_index=True)


          # df.to_csv("evaluation_results.csv", index=False)
          # print('evaluation appended')
     

    


if __name__ == "__main__":
    main()