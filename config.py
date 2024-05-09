from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import os
llamaapi=os.environ["LLAMA_CLOUD_API_KEY"] = "enter your llamaparse api key"
openai_key="enter your open ai api key"
OPENAI_API_KEY=openai_key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
llm=ChatOpenAI()
embedding=OpenAIEmbeddings()

