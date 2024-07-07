from llama_index import SummaryIndex, Document

index = SummaryIndex([])
text_chunks = ['text_chunk_1', 'text_chunk_2', 'text_chunk_3']

doc_chunks = []
for i, text in enumerate(text_chunks):
    doc = Document(text=text, id_=f"doc_id_{i}")
    doc_chunks.append(doc)

print(doc_chunks)

# insert
for doc_chunk in doc_chunks:
    index.insert(doc_chunk)


# delete documents from index
index.delete_ref_doc("doc_id_0", delete_from_docstore=True)



# update documents 
doc_chunks[0].text = "Brand new document text"
index.update_ref_doc(
    doc_chunks[0],
    update_kwargs={"delete_kwargs": {'delete_from_docstore': True}}
)
print(doc_chunks)


#refresh index 
# modify first document, with the same doc_id
doc_chunks[0] = Document(text='Super old document text', id_="doc_id_0")

# add a new document
doc_chunks.append(Document(text="This isn't in the index yet, but it will be soon!", id_="doc_id_3"))

print(doc_chunks)

# refresh the index
refreshed_docs = index.refresh_ref_docs(
    doc_chunks,
    update_kwargs={}
)

print(refreshed_docs)
