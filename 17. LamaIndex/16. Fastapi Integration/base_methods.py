from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage
import os

os.environ["OPENAI_API_KEY"] = "sk-*************"
index = None

def initialize_index():
    global index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist()


def query_index(query):
    global index
    if index is None :
        storage_context = StorageContext.from_defaults(persist_dir= "./storage")
        index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return response


