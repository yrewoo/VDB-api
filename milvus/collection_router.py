from fastapi import UploadFile
from milvus.milvus_router import MilvusDB
from milvus.leetcode import process_file as process_lc_solution
from milvus.grepp import process_file as process_grepp_solution

async def insert_data(file: UploadFile, collection_name: str):
    milvusdb = MilvusDB()
    existing_collections = milvusdb.list_collections()  

    if collection_name not in existing_collections:
        raise Exception(f"Collection {collection_name} does not exist.")
    
    if collection_name == "leetcode_solution":
        try:
            await process_lc_solution(file, collection_name)
            return {
                "collection": collection_name,
                "result": "success"
            }
        except Exception as e:
            return {"error": str(e)}
        
    if collection_name == "grepp" or collection_name == "grepp_solution":
        try:
            await process_grepp_solution(file, collection_name)
            return {
                "collection": collection_name,
                "result": "success"
            }
        except Exception as e:
            return {"error": str(e)}