import csv
import json

from src.registry import registry
from src.util.logger import logger
from src.milvus_router import MilvusDB
from fastapi import APIRouter, File, UploadFile, BackgroundTasks


router = APIRouter()
registry.load_providers()
milvus_client = MilvusDB()

upload_status = {}

def parse_filetype(file_content: str, file_type: str):
    try:
        if file_type == "json":
            return json.loads(file_content)  # ✅ JSON 파싱
        elif file_type == "jsonl":
            return [json.loads(line) for line in file_content.strip().split("\n")]  # ✅ JSONL 파싱
        elif file_type == "csv":
            reader = csv.DictReader(file_content.splitlines())  # ✅ CSV 파싱
            return [row for row in reader]
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        logger.error(f"❌ Error parsing {file_type} file: {e}")
        raise ValueError(f"Invalid {file_type} format")

# ✅ 백그라운드에서 실행할 데이터 업로드 함수
def process_upload(collection_name: str, file_content: str, file_type: str):
    try:
        provider = registry.get_provider(f"{collection_name}_provider")
        if not provider:
            logger.error(f"❌ Provider {collection_name}_provider not found")
            return
        
        upload_status[collection_name] = "processing"
        contents = parse_filetype(file_content=file_content,
                                  file_type=file_type)
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
        
        filename = file.filename.lower()
        print(filename)
        if filename.endswith(".json"):
            file_type = "json"
        elif filename.endswith(".jsonl"):
            file_type = "jsonl"
        elif filename.endswith(".csv"):
            file_type = "csv"
        else:
            return {"error": "Unsupported file type. Please upload a JSON, JSONL, or CSV file."}

        background_tasks.add_task(process_upload, 
                                  collection_name, 
                                  file_content,
                                  file_type)
        return {"message": "Upload started in background"}

    except Exception as e:
        logger.error(e)
        return {"error": str(e)}
    
@router.get("/upload_status/")
async def get_upload_status(collection_name: str):
    status = upload_status.get(collection_name, "not_started")
    return {"collection_name": collection_name, "status": status}