import os
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from src.util.logger import logger
from src.config import EMBEDDING_MODEL, MAX_TOKENS
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

load_dotenv(override=True)
API_KEY = os.getenv("OPENAI_API_KEY")

class EmbeddingGenerator:
    def __init__(self):
        if EMBEDDING_MODEL == "text-embedding-3-small":
            self.model = OpenAI(api_key = API_KEY)
            self.tokenizer = tiktoken.encoding_for_model(EMBEDDING_MODEL)
        else:
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

    def truncate_to_tokens(self, text, max_tokens=MAX_TOKENS[EMBEDDING_MODEL]):
        """텍스트를 모델의 최대 토큰 길이에 맞게 자름"""
        try:
            if EMBEDDING_MODEL == "text-embedding-3-small":
                tokens = self.tokenizer.encode(text)
                truncated_tokens = tokens[:max_tokens]
                return self.tokenizer.decode(truncated_tokens)
            else:
                tokenized = self.tokenizer(text, truncation=True, max_length=max_tokens)
                return self.tokenizer.decode(tokenized["input_ids"])
        except Exception as e:
            logger.error(f"❌ Error in truncate_to_tokens: {e}")
        
    def embed(self, text):
        text = self.truncate_to_tokens(text)
        if EMBEDDING_MODEL == "text-embedding-3-small":
            response = self.model.embeddings.create(input=text, model=EMBEDDING_MODEL)
            return [x.embedding for x in response.data]
        else:
            return [self.model.encode(text).tolist()]

embedder = EmbeddingGenerator()
