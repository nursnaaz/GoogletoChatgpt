import streamlit as st
import pickle
from utility import preprocess_text

with open('index.pkl', 'rb') as f:
    index = pickle.load(f)
    
with open('document_details.pkl', 'rb') as f:
    document_details = pickle.load(f)

def search(query, index):
    document_result = []
    query_preprocess = preprocess_text(query)
    results = set()
    for word in query_preprocess.split(' '):
        if word in index:
            if not results:
                results = index[word]
            else:
                results = results.union(index[word])
        else:
            set()
    if results:
        results = list(results)
        results = results[:20]
        for doc_id in results:
            doc = document_details[doc_id]   
            document_result.append([doc['title'],doc['url'],doc['abstract'] ])
    else:
        document_result
    return document_result

st.title('Simple Search Engine')

query = st.text_input('Enter your search query')

if query:
    result = search(query, index)
    if result:
        for doc in result:
            st.markdown(f'### {doc[0]}')
            st.markdown(f'### {doc[1]}')
            st.write('Abstract : ',doc[2])
    else:
        st.write('No Result')
            
    





