import base64
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
from faiss import write_index,read_index
import faiss
import numpy as np
import pickle
load_dotenv() 
openai_key=os.getenv("openai_key")
from openai import OpenAI

client=OpenAI(api_key=openai_key)

#get embeddings
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )

    return response.data
#get file path for file (helps in path errors)
def get_data_file_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)

#create embeddings and write index to local file
def create_embeddings_write_index():
    try:
        df = pd.read_csv("datasource.csv")
        embeddings = get_embedding(df['user_input'].tolist())
        embeddedlist = [embedding['embedding'] for embedding in embeddings]
        question_embeddings = np.array(embeddedlist, dtype='float32')
        
        # Save embeddings
        with open('question_embeddings.pkl', 'wb') as f:
            pickle.dump(question_embeddings, f)
        
        # Create and save FAISS index
        dimension = question_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(question_embeddings)
        faiss.write_index(index, "index.bin")
        
        print('Created and saved FAISS index.')
    except Exception as e:
        print(f"An error occurred while creating embeddings: {e}")


@st.cache_resource
def load_model_and_embeddings():
    with open('question_embeddings.pkl', 'rb') as f:
        question_embeddings = pickle.load(f)
    return question_embeddings


#use semantic search to find answers
def search_with_faiss(prompt):
    df_file_path=get_data_file_path('datasource.csv')
    df = pd.read_csv(df_file_path)

    index_file_path=get_data_file_path('index.bin')
    index = faiss.read_index(index_file_path)

    # Get query embedding
    query_embedding = get_embedding([prompt])
    emblist = [embedding.embedding for embedding in query_embedding]
    query_embedding = np.array(emblist, dtype='float32')
    
    # Check embedding dimensions
    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"Index embedding shape: {index.d}")

    # Perform FAISS search
    D, I = index.search(query_embedding, k=1)
    print(f"Distances: {D}, Indices: {I}")

    best_match_idx = I[0][0]
    best_match_score = D[0][0]
    best_match_score=1-best_match_score
    # Retrieve the best matching row
    best_match_row = df.iloc[best_match_idx]
    response = best_match_row['llm_output']
    video_path = best_match_row['video_path']
    
    return response, video_path, best_match_score

# Example usage
# response, video_path, score = search_with_faiss("What is VSC?")
# print(f"Response: {response}, Video Path: {video_path}, Score: {score}")

#used to display the video on streamlit app
def get_base64_video(video_path):
            with open(video_path, "rb") as file:
                video_data = file.read()
                print('base64 video path',video_path)
                encoded_video = base64.b64encode(video_data).decode("utf-8")
            return encoded_video


