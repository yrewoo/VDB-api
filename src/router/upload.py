import json
from tqdm import tqdm
from src.util.logger import logger
from src.milvus_router import MilvusDB
from src.registry import registry

from fastapi import APIRouter, File, UploadFile, BackgroundTasks

router = APIRouter()
registry.load_providers()
milvus_client = MilvusDB()

upload_status = {}

# ✅ 백그라운드에서 실행할 데이터 업로드 함수
def process_upload(collection_name: str, file_content: str):
    try:
        provider = registry.get_provider(f"{collection_name}_provider")
        if not provider:
            logger.error(f"❌ Provider {collection_name}_provider not found")
            return
        
        upload_status[collection_name] = "processing"
        contents = json.loads(file_content)
        provider.parse_data(contents)
        upload_status[collection_name] = "completed"

        logger.info(f"Upload completed for collection: {collection_name}")

    except Exception as e:
        logger.error(f"❌ Error in background task: {e}")
        upload_status[collection_name] = "failed"


@router.post("/upload/")
async def upload_file(
    collection_name: str, 
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()):
    try:
        file_content = await file.read()
        file_content = file_content.decode("utf-8")
        background_tasks.add_task(process_upload, collection_name, file_content)
        return {"message": "Upload started in background"}

    except Exception as e:
        logger.error(e)
        return {"error": str(e)}
    
@router.get("/upload_status/")
async def get_upload_status(collection_name: str):
    status = upload_status.get(collection_name, "not_started")
    return {"collection_name": collection_name, "status": status}