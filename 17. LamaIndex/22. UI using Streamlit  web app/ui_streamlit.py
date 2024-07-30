import os, streamlit as st
import openai
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.openai import OpenAI
import traceback
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)


# Define a simple Streamlit app
st.title("Ask LlamaIndex")
query = st.text_input("What would you like to ask?")

# If the 'Submit' button is clicked
if st.button("Submit"):
    if not query.strip():
        st.error(f"Please provide the search query.")
    else:
        try:
                # define custom LLM model
                llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")

                # Load documents from the 'data' directory
                documents = SimpleDirectoryReader('data').load_data()
                service_context = ServiceContext.from_defaults(llm=llm)

                # build the index
                index = VectorStoreIndex.from_documents(documents, service_context=service_context)

                query_engine= index.as_query_engine()
                response = query_engine.query(query)
                st.success(response)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error(traceback.format_exc())



