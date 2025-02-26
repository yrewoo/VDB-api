from src.registry import registry
from src.util.logger import logger
from src.milvus_router import MilvusDB

from fastapi import APIRouter, Query

router = APIRouter()
milvus_client = MilvusDB()

@router.get("/expr_search/")
async def expr_search(
    collection_name: str, 
    expr: str = Query(example="problem_id == 1"),
    limit: int = Query(default=1)):
    try:
        provider = registry.get_provider(f"{collection_name}_provider")
        if not provider:
            logger.error(f"❌ Provider {collection_name}_provider not found")
            return
        
        output_fields = provider.get_output_fields()
        results = milvus_client.query(collection_name=collection_name, 
                                           output_fields=output_fields,
                                           expr=expr, 
                                           limit=limit)
        return {"total": len(results), "results": results}
    except Exception as e:
        logger.error(e)
        return {"error": str(e)}