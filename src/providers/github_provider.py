# Github Provider
import os
import json
from tqdm import tqdm
from src.config import DIMENSION
from src.util.logger import logger
from src.milvus_router import MilvusDB
from pymilvus import FieldSchema, DataType
from src.util.embedding_model import embedder
from src.providers.base_provider import BaseProvider
from src.util.existing_checker import get_existing_solution_ids

milvusdb = MilvusDB()


class GithubProvider(BaseProvider):
    def __init__(self):
        super().__init__(collection_name="github", uid_field="id")
        
    def get_schema(self):
        fields = [
            FieldSchema(name='id', dtype=DataType.VARCHAR, max_length=10000, is_primary=True),
            FieldSchema(name='file_name', dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name='line_count', dtype=DataType.INT64),
            FieldSchema(name='mark', dtype=DataType.FLOAT),
            FieldSchema(name='code', dtype=DataType.VARCHAR, max_length=64000),
            FieldSchema(name='query_nsx', dtype=DataType.VARCHAR, max_length=65500),
            FieldSchema(name='query_nlx', dtype=DataType.VARCHAR, max_length=65500),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        ]
        embed_field = 'embedding'
        return fields, embed_field

    def get_output_fields(self):
        return ["id", "file_name", "line_count", "mark", "code", "query_nsx", "query_nlx"]
    
    def parse_data(self, json_data):
        collection = milvusdb.connect_collection(self.collection_name)
        existing_ids = get_existing_solution_ids(
            collection, 
            self.uid_field, 
            is_int=False)
        logger.info(f"[dedupe] loaded existing_ids: {len(existing_ids)}")
        
        with tqdm(total=len(json_data)) as pbar:
            logger.info("Start parsing data...")
            for element in json_data:
                if element["id"] in existing_ids:
                    pbar.set_description(f"Skipping {element['id']}")
                    logger.info(f"Skipping {element['id']}")
                    pbar.update(1)
                    continue

                pbar.set_description(f"Embedding {element['id']}")
                logger.info(f"Embedding {element['id']}")
                cut_query_nsx = embedder.truncate_to_max_length(element["query_nsx"])
                cut_query_nlx = embedder.truncate_to_max_length(element["query_nlx"])
                merged_content = element["file_name"] + element["code"] + element["query_nsx"] + element["query_nlx"]
                cut_content = embedder.truncate_to_tokens(merged_content)  # 최대 토큰 길이로 자르기
                array_data = [
                    [element["id"]],
                    [element["file_name"]],
                    [element["line_count"]],
                    [element["mark"]],
                    [element["code"]],
                    [cut_query_nsx],
                    [cut_query_nlx],
                    embedder.embed(cut_content)
                ]
                milvusdb.ingest(collection, array_data)
                pbar.update(1)
        logger.info("End parsing data...")
    
__all__ = [
    "GithubProvider"
]
