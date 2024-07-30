# packages to install 
    # pip install fastapi
    # pip install uvicorn[standard]

from fastapi import FastAPI

app = FastAPI()

@app.get("/test")
async def root():
    return {"message" : "Hi How are you doing?"}