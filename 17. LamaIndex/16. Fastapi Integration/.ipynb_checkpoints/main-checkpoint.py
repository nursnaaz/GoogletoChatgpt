from fastapi import FastAPI
from pydantic import BaseModel
from base_methods import initialize_index, query_index
from contextlib import asynccontextmanager

# define lifespan to load index before it can handle queries
@asynccontextmanager
async def lifespan(app: FastAPI):
    # load index
    initialize_index()
    yield

# define FastAPI instance
app = FastAPI(lifespan=lifespan)


# defines the request model
class Query(BaseModel):
    text : str


# defines POST endpoint
@app.post("/query")
async def get_response(query : Query):
    if query.text is None : 
        return "No query is provided, please pass a query."
    output = query_index(query.text)
    response = output.response
    return {"response " : response}







