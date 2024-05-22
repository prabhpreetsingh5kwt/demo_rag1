import os
import openai
import pandas as pd
import streamlit as st 
from config import openai_key,langchain_api_key
from app_utils import pdf_qa
from langchain_openai import OpenAIEmbeddings
from uptrain import EvalLLM, Evals
from vector_store import create_index
from eval_utils import generate_llm_response

os.environ["OPENAI_API_KEY"] = openai_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ['LANGCHAIN_PROJECT'] = "Versa" 


client = openai.OpenAI()

embeddings = OpenAIEmbeddings()

eval_llm = EvalLLM(openai_api_key=openai_key)

vectorstore,retriever=create_index()

def main():

     st.title("Chat with Versa Using Langchain!")
     # On Every reload:
     if "messages" not in st.session_state:
          st.session_state.messages=[]

          # Clear conversational buffer memory
          pdf_qa.memory.clear()

          # Initiate Conversation and append it
          response=pdf_qa({"question": "Hello Versa!"})
          response=response['answer']
          st.session_state.messages.append({"role": "assistant", "content": str(response)})


     # Prints chat message history in streamlit ui
     for message in st.session_state.messages:
          with st.chat_message(message['role']):
               st.markdown(message["content"])
     

     # Enters chat prompt and checks if it is not NONE 
     if prompt := st.chat_input("Feel free to ask about your insurance policies!"):
          # Display user message in chat style
          st.chat_message("user").markdown(prompt)
          

          # Add user message to chat history
          st.session_state.messages.append({"role": "user", "content": prompt})


          # Get response from qa chain
          response=pdf_qa({"question": prompt})
          response=response['answer']


          # Write response
          with st.chat_message("assistant"):
               st.write(response)


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