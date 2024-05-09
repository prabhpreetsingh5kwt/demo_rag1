from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_parse import LlamaParse
from config import openai_key

parser = LlamaParse(result_type="markdown",verbose=True,)
file_extractor = {".pdf": parser}

embed_model=HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print(embed_model)
Settings.embed_model =embed_model

OPENAI_API_KEY=openai_key
import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


try:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    # load index
    index = load_index_from_storage(storage_context)
    print('old storage')
except:
    print('creating new storage')
    documents = LlamaParse(result_type="markdown").load_data(r"C:\Users\prabh\OneDrive\Desktop\menuverse\demo_rag1\data\Versa_KnowledgeBase.docx")
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist()
    print("saved storage")
    

memory=ChatMemoryBuffer(token_limit=1500)
chat_engine=index.as_chat_engine(verbose=True,memory=memory,system_prompt=( """Your name is Versa, a Virtual Assistant. 
    you have to cater user in very friendly tone.Your are a very wise and expert at selling.
    your intention is to sell warranty plans considering user's requirements. 
    Do not convey your intentions to user. for first time, greet user by their first name i.e 'kito'"""),)


