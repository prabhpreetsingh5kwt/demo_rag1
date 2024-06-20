import base64
import streamlit as st
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import os
import time
import faiss
import numpy as np
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
 
        model = SentenceTransformer('all-MiniLM-L6-v2')
        with open('question_embeddings.pkl', 'rb') as f:
            question_embeddings = pickle.load(f)
        df = pd.read_csv("datasource.csv")
        questions = df['user_input'].tolist()
        return model, question_embeddings, df,questions
   



def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # Dimension of the embeddings
    index = faiss.IndexFlatL2(dimension)  # L2 distance
    index.add(embeddings)
    return index


def search_with_faiss(prompt):
    model, question_embeddings, df, questions = load_model_and_embeddings()
    index = create_faiss_index(np.array(question_embeddings))

    query_embedding = model.encode([prompt])
    D, I = index.search(query_embedding, k=1)  # Search for the nearest neighbor
    best_match_idx = I[0][0]
    best_match_score = D[0][0]
    
    # Retrieve the best matching row
    best_match_row = df.iloc[best_match_idx]
    response = best_match_row['llm_output']
    video_path = best_match_row['video_path']
    return response, video_path, best_match_score


# Load model and embeddings
# model, question_embeddings, df, questions = load_model_and_embeddings()

# Create FAISS index
# index = create_faiss_index(np.array(question_embeddings))


# prompt = "What are the things I should know about windshield protection?"
# response, video_path, similarity_score = search_with_faiss(prompt)

# print(f"Response: {response}")
# print(f"Video Path: {video_path}")
# print(f"Similarity Score: {similarity_score}")





# #semantic similarity of questions and user
# def match_datasource(prompt):
#     model, question_embeddings, df = load_model_and_embeddings()

#     query_embedding = model.encode(prompt, convert_to_tensor=True)
#     similarities = util.pytorch_cos_sim(query_embedding, question_embeddings)
#     best_match_idx = similarities.argmax().item()  # Convert tensor to integer
#     best_match_score = similarities[0][best_match_idx].item()  # Get the similarity score
#     best_match_row = df.iloc[best_match_idx]
#     response = best_match_row['llm_output']
#     video_path = best_match_row['video_path']
#     return response, video_path, best_match_score


def get_base64_video(video_path):
            with open(video_path, "rb") as file:
                video_data = file.read()
                print('base64 video path',video_path)
                encoded_video = base64.b64encode(video_data).decode("utf-8")
            return encoded_video



