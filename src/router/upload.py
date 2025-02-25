import json
from tqdm import tqdm
from src.util.logger import logger
from src.milvus_router import MilvusDB
from src.registry import ProviderRegistry

from fastapi import APIRouter, File, UploadFile

router = APIRouter()
milvus_client = MilvusDB()
registry = ProviderRegistry()
registry.load_providers()

@router.post("/upload/")
async def upload_file(collection_name: str, file: UploadFile = File(...)):
    try:
        provider = registry.get_provider(f"{collection_name}_provider")
        print(provider)
        if not provider:
            return {"error": "Provider not found"}
        
        contents = json.loads(file.file.read().decode("utf-8"))
        _ = provider.parse_data(contents)

        return {"message": "File uploaded successfully"}

    except Exception as e:
        logger.error(e)
        return {"error": str(e)}