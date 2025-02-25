from fastapi import UploadFile
from milvus.milvus_router import MilvusDB
from milvus.leetcode import process_file as process_lc_solution
from milvus.grepp import process_file as process_grepp_solution

async def insert_data(file: UploadFile, collection_name: str):
    milvusdb = MilvusDB()
    existing_collections = milvusdb.list_collections()  

    if collection_name not in existing_collections:
        raise Exception(f"Collection {collection_name} does not exist.")
    
    try:
        if collection_name == "leetcode_solution":
            await process_lc_solution(file, collection_name)
            return {
                "collection": collection_name,
                "result": "success"
            }
            
        if collection_name == "grepp" or collection_name == "grepp_solution":
            await process_grepp_solution(file, collection_name)
            return {
                "collection": collection_name,
                "result": "success"
            }
    except Exception as e:
        return {"error": str(e)}
    
async def query(collection_name: str, expr: str, limit):
    milvus = MilvusDB()
    existing_collections = milvus.list_collections()  
    if collection_name not in existing_collections:
        raise Exception(f"Collection {collection_name} does not exist.")
    
    collection = milvus.connect_collection(collection_name)
    try:
        if collection_name == "leetcode_solution":
            fields = ["problem_id", "solution_id", "description", "solution"]
        if collection_name == "grepp":
            fields = ["problem_id", "title", "partTitle", "languages", "level", "description", "testcases"]
        if collection_name == "grepp_solution":
            fields = ["solution_id", "problem_id", "language", "code"]

        result = milvus.query(collection, fields, expr, limit)
        total = len(result)
        return {"total": total, "result": result}
    except Exception as e:
        return {"error": str(e)}
            



async def search(collection_name: str, text: str, limit: int):
    milvus = MilvusDB()
    existing_collections = milvus.list_collections()  
    if collection_name not in existing_collections:
        raise Exception(f"Collection {collection_name} does not exist.")
    
    collection = milvus.connect_collection(collection_name)
    print(collection)
    try:
        if collection_name == "leetcode_solution":
            fields = ["problem_id", "solution_id", "description", "solution"]
        if collection_name == "grepp":
            fields = ["problem_id", "title", "partTitle", "languages", "level", "description", "testcases"]
        if collection_name == "grepp_solution":
            fields = ["solution_id", "problem_id", "language", "code"]

        result = milvus.search(collection=collection, data=text, target='embedding', output_fields=fields, top_k=limit)
        total = len(result)
        return {"total": total, "result": result}
    except Exception as e:
        return {"error": str(e)}
