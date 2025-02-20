import uvicorn

from typing import List
from fastapi import FastAPI, File, UploadFile
from util.logger import logger
from milvus.collection_router import insert_data

app = FastAPI()

@app.post("/upload/")
async def upload_file(collection_name: str, file: UploadFile = File(...)):
    try:
        result = await insert_data(file, collection_name)
        return result
    except Exception as e:
        logger.error(e)
        return {"error": str(e)}

@app.get("/search_expr/")
async def search_expr(collection_name: str, expr: str):
    pass

@app.post("/vector_search/")
async def vector_search(collection_name: str, query: str):
    pass

if __name__ == "__main__":
    uvicorn.run("main:app", 
                host="0.0.0.0", 
                port=10001, 
                reload=True,)