import os
import openai
import chainlit as cl
from llama_parse import LlamaParse
from config import openai_key
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.callbacks import CallbackManager
from llama_index.core.service_context import ServiceContext
OPENAI_API_KEY=openai_key
import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
try:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    # load index
    index = load_index_from_storage(storage_context)
except:
    documents = LlamaParse(result_type="markdown").load_data(r"C:\Users\prabh\OneDrive\Desktop\menuverse\demo_rag1\data\Z2 Service Contract.docx")
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist()


@cl.on_chat_start
async def start():
    Settings.llm = OpenAI(
        model="gpt-3.5-turbo", temperature=0.1, max_tokens=1024, streaming=True
    )
    embed_model=HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
    Settings.embed_model = embed_model
    # Settings.context_window = 4096

    service_context = ServiceContext.from_defaults(callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]))
    query_engine = index.as_chat_engine(streaming=True, similarity_top_k=2, service_context=service_context,
                                        system_prompt=( """Your name is Versa, a Virtual Assistant. 
    you have to cater user in very friendly tone.Your are a very wise and expert at selling.
    your intention is to sell warranty plans considering user's requirements. 
    Do not convey your intentions to user. for first time, greet user by their first name i.e 'kito'"""))
    cl.user_session.set("query_engine", query_engine)

    await cl.Message(
        author="Versa", content="Hello! Im Versa, your Virtual AI Assistant. How may I help you?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine") # type: RetrieverQueryEngine

    msg = cl.Message(content="", author="Assistant")

    res = await cl.make_async(query_engine.query)(message.content)
    ans=str(res)
    print("res",res)
    print('res_type,',type(res))

    for token in ans:
        await msg.stream_token(token)
    await msg.send()
