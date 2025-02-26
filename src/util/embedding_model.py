import os
from openai import OpenAI
from dotenv import load_dotenv
from src.config import EMBEDDING_MODEL
from sentence_transformers import SentenceTransformer

load_dotenv(override=True)
API_KEY = os.getenv("OPENAI_API_KEY")

class EmbeddingGenerator:
    def __init__(self):
        if EMBEDDING_MODEL == "text-embedding-3-small":
            self.model = OpenAI(api_key = API_KEY)
        else:
            self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed(self, text):
        if EMBEDDING_MODEL == "text-embedding-3-small":
            response = self.model.embeddings.create(input=text, model=EMBEDDING_MODEL)
            return [x.embedding for x in response.data]
        else:
            return [self.model.encode(text).tolist()]

embedder = EmbeddingGenerator()
