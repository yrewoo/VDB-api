import os
import tiktoken

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL") 
MAX_TOKENS = 8190  # 최대 토큰 길이 제한

def truncate_to_tokens(text, max_tokens=MAX_TOKENS):
    enc = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    tokens = enc.encode(text)
    truncated_tokens = tokens[:max_tokens]
    return enc.decode(truncated_tokens)