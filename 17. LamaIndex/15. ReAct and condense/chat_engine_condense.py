from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

def initialize_index(index_dir):
    global index
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)

    docstore_path = os.path.join(index_dir, 'docstore.json')
    
    if os.path.exists(docstore_path):
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        index = load_index_from_storage(storage_context)
    else:
        llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
        service_context = ServiceContext.from_defaults(llm=llm)
        documents = SimpleDirectoryReader("input/text").load_data()
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        index.storage_context.persist(persist_dir=index_dir)


def chat(query):
# build chat engine from index
    chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
    response = chat_engine.chat(query)
    print(response)

def react_chat(query):
    chat_engine = index.as_chat_engine(chat_mode="react", verbose=True)
    response = chat_engine.chat(query)
    print(response)

initialize_index("./storage")
