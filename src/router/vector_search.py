from src.util.logger import logger
from src.milvus_router import MilvusDB
from src.registry import registry
from fastapi import APIRouter, Query

router = APIRouter()
milvus_client = MilvusDB()

@router.get("/vector_search/")
async def vector_search(
    collection_name: str, 
    expr: str = Query(example="problem_id == 1"),
    limit: int = Query(default=1)):
    try:
        result = await query(collection_name, expr, limit)
        return result
    except Exception as e:
        logger.error(e)
        return {"error": str(e)}