import uvicorn

from typing import List
from util.logger import logger
from fastapi import FastAPI, File, UploadFile, Query
from milvus.collection_router import insert_data, query, search

app = FastAPI()

@app.post("/upload/")
async def upload_file(collection_name: str, file: UploadFile = File(...)):
    try:
        result = await insert_data(file, collection_name)
        return result
    except Exception as e:
        logger.error(e)
        return {"error": str(e)}

@app.get("/expr_search/")
async def expr_search(
    collection_name: str, 
    expr: str = Query(example="problem_id == 1"),
    limit: int = Query(default=1)):
    try:
        result = await query(collection_name, expr, limit)
        return result
    except Exception as e:
        logger.error(e)
        return {"error": str(e)}

@app.post("/vector_search/")
async def vector_search(collection_name: str, text: str):
    try:
        result = await search(collection_name, text)
        return result
    except Exception as e:
        logger.error(e)
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", 
                host="0.0.0.0", 
                port=10001, 
                reload=True,)