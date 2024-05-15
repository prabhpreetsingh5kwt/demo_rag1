from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import os
llamaapi=os.environ["LLAMA_CLOUD_API_KEY"] = "llx-"
openai_key="s"
OPENAI_API_KEY=openai_key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
llm=ChatOpenAI()
embedding=OpenAIEmbeddings()

