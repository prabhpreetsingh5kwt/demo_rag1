# Responsible for evaluation
import os
from config import openai_key

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
