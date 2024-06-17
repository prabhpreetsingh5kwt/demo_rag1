import base64
import streamlit as st
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import os
import time
import pickle
load_dotenv() 
openai_key=os.getenv("openai_key")


def create_embeddings():
    try:
        df = pd.read_csv("datasource.csv")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        question_embeddings = model.encode(df['user_input'].tolist(), convert_to_tensor=True)
        
        with open('question_embeddings.pkl', 'wb') as f:
            pickle.dump(question_embeddings, f)
            
    except Exception as e:
        print(f"An error occurred while creating embeddings: {e}")


@st.cache_resource
def load_model_and_embeddings():
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        with open('question_embeddings.pkl', 'rb') as f:
            question_embeddings = pickle.load(f)
        df = pd.read_csv("datasource.csv")
        return model, question_embeddings, df
    except Exception as e:
         print(f'error in loading model or ques embeddings{e}')


#semantic similarity of questions and user query
def match_datasource(prompt):
    model, question_embeddings, df = load_model_and_embeddings()

    query_embedding = model.encode(prompt, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, question_embeddings)
    best_match_idx = similarities.argmax().item()  # Convert tensor to integer
    best_match_row = df.iloc[best_match_idx]
    response = best_match_row['llm_output']
    video_path = best_match_row['video_path']
    return response, video_path


def get_base64_video(video_path):
            with open(video_path, "rb") as file:
                video_data = file.read()
                print('base64 video path',video_path)
                encoded_video = base64.b64encode(video_data).decode("utf-8")
            return encoded_video



