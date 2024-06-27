import base64
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import time
load_dotenv() 
from semantic_srch_utils import search_with_faiss,get_base64_video
openai_key=os.getenv("openai_key")

# Streamlit app
def main():
    if "messages" not in st.session_state:
        st.session_state.messages=[]

        st.session_state.messages.append({"role": "assistant", "content": 'Hi, what can i help you with today?'})

    st.cache_data()
    # This function loads the image and converts it into readable format
    def get_base64_image(image_path):
        with open(image_path, "rb") as file:
            image_data = file.read()
            encoded_image = base64.b64encode(image_data).decode("utf-8")
        return encoded_image


    link_url = "versa.png"
    image_base64 = get_base64_image(link_url)
    

    # HTML to insert in streamlit 
    image_html = f"""
        <a href="data:image/png;base64,{image_base64}" target="_blank">
        <img src="data:image/png;base64,{image_base64}" alt="Image" style='position: fixed;bottom: 113px;
        right: 87px;
        height: 120px;
        width: 120px ;
        border-radius: 50%;
        z-index: 11;
        object-fit:cover;
        '>
        </a>
        """
    
    # Show static image of Versa
    st.markdown(image_html, unsafe_allow_html=True)


    # Chat application
    st.title("Chat with Versa Using Langchain!")

    for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.markdown(message["content"])

    if prompt:= st.chat_input("Feel free to ask about your insurance policies!"):
        st.chat_message('user').markdown(prompt)
        start_time = time.time()
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
    
        # Semantic match to get response and video path for query 
        response,video_path,score=search_with_faiss(prompt)
        st.session_state.messages.append({"role": "assistant", "content": str(response)})
        print(video_path)

        #show video
        video_base64 = get_base64_video(video_path)
        video_html = f"""
        <video controls autoplay  style='position: fixed;bottom: 113px;
            right: 87px;
            height: 120px;
            width: 120px;
            border-radius: 50%;
            z-index: 11;
            object-fit: cover;'>
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        """
        st.markdown(video_html, unsafe_allow_html=True)

        video_path=''
        # Calculate Processing Time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken: {elapsed_time:.2f} seconds")
        st.write(f"Time taken: {elapsed_time:.2f} seconds")
        def char_response_generator(response):
                for char in response:
                        yield char
                        time.sleep(0.001)
        

        # Write response
        st.write(score)
        with st.chat_message("assistant"):
            st.write_stream(char_response_generator(response))
if __name__ == "__main__":
    main()


