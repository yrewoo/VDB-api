from src.util.logger import logger
from src.milvus_router import MilvusDB
from src.registry import registry
from fastapi import APIRouter, Query

router = APIRouter()
milvus_client = MilvusDB()

@router.get("/vector_search/")
async def vector_search(
    collection_name: str, 
    text: str,
    limit: int = Query(default=1),
    expr: str = Query(default=None, description="Filtering expression")):
    try:
        provider = registry.get_provider(f"{collection_name}_provider")
        if not provider:
            logger.error(f"❌ Provider {collection_name}_provider not found")
            return
        
        _, embed_field = provider.get_schema()
        output_fields = provider.get_output_fields()
        results = milvus_client.search(collection_name=collection_name,
                                      data=text,
                                      target_field=embed_field,
                                      output_fields=output_fields,
                                      top_k=limit,
                                      expr=expr)
        return {"total": len(results), "results": results}
    except Exception as e:
        logger.error(e)
        return {"error": str(e)}