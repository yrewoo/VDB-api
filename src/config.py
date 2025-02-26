import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 🔹 사용 모델 설정
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# 🔹 모델별 차원 설정
MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/all-MiniLM-L12-v2": 384
}

MAX_TOKENS = {
    "text-embedding-3-small": 8192,
    "sentence-transformers/all-mpnet-base-v2": 512,
    "sentence-transformers/all-MiniLM-L12-v2": 256
}

# 🔹 선택한 모델에 따른 차원 자동 설정
DIMENSION = MODEL_DIMENSIONS.get(EMBEDDING_MODEL)

print(f"✅ Using embedding model: {EMBEDDING_MODEL} (DIMENSION={DIMENSION})")
