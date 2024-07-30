import os
import openai
from langchain.chat_models import ChatOpenAI
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, LLMPredictor

# step 1 : pass streaming as true in LLMPredictor
llm_predictor = LLMPredictor(ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True))

documents = SimpleDirectoryReader("input/text").load_data()
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
index = VectorStoreIndex.from_documents(documents= documents, service_context=service_context)

# step 2 : Pass streaming as true in query engine
query_engine = index.as_query_engine(streaming = True)

# ask questions
response = query_engine.query("What did author do growing up?")

# print stream response
response.print_response_stream()