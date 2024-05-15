import pandas
import json
import os
import streamlit as st
from config import openai_key
from raglang_utils import vectorstore
from uptrain import EvalLLM, Evals

os.environ['OPENAI_API_KEY'] = openai_key
import openai

client = openai.OpenAI()


def generate_llm_response(question, vectorstore):
    documents = vectorstore.similarity_search(question, k=1)
    context = " , ".join([x.page_content for x in documents])

    prompt = f"""
        Answer the following user query using the retrieved document in less than 3 sentences:
        {question}
        The retrieved document has the following text:
        {context}

        Answer:
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=[{"role": "system", "content": prompt}], temperature=0.1
    ).choices[0].message.content

    return [{'question': question, 'context': context, 'response': response}]

questions = [
    "What is gap?",
    'What is windshield protection?',
    'What is theft deterrent?',
    'will my VCS work if i leave US and go to mexico?',
    "at most what amount can i get with gap?",
    'will windshield protection cover scratches?',
    'what will happen if my car is broken into?',
    'What if someone tries to break my car?',
    "if i am stuck on a highway, and engine starts to make weird noises,what will you do to help me?",
    'what happens if my car gets stolen? how will your product help me?',
    "do you cover powertrain parts?"
    'what about electricals in my car?',
    "what's the catch?"
]

results = []
for question in questions:
    with st.spinner("generating responses..."):
        results.extend(generate_llm_response(question, vectorstore))