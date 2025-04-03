from fastapi import FastAPI
from src.milvus_router import MilvusDB
from src.registry import registry
from src.router.upload import router as upload_router
from src.router.expr_search import router as expr_search_router
from src.router.vector_search import router as vecotr_search_router

app = FastAPI()
# Milvus 클라이언트 초기화
milvus_client = MilvusDB()

registry.load_providers()

for provider_name, provider in registry.providers.items():
    if provider_name != "base_provider":
        schema, embed_field = provider.get_schema()
        collection_name = provider_name.replace("_provider", "")
        print(f"🛠 Creating collection: {collection_name}")
        milvus_client.create_collection(collection_name, schema, embed_field, drop_existing=False)

# 라우터 등록
app.include_router(upload_router, prefix="/api")
app.include_router(expr_search_router, prefix="/api")
app.include_router(vecotr_search_router, prefix="/api")

@app.get("/")
def root():
    return {"message": "Welcome to Milvus Code Indexer"}

