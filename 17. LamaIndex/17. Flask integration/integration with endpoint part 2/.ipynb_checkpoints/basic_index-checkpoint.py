import os
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, StorageContext, load_index_from_storage
from llama_index import LLMPredictor, ServiceContext
from langchain.chat_models import ChatOpenAI
import configparser
from flask import Flask, request

config = configparser.ConfigParser()
config.read('config.ini')

# get openai api key
os.environ['OPENAI_API_KEY'] = config["openai"]["api_key"]


index = None

app = Flask(__name__)

def initialize_index(index_dir):
    global index
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    if os.path.exists(index_dir):
        index = load_index_from_storage(storage_context)
    else:
        llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
        documents = SimpleDirectoryReader("input/text").load_data()
        index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context, storage_context=storage_context)
        index.storage_context.persist(index_dir)

if __name__ == "__main__":
    print("initializng index")
    initialize_index("index")
