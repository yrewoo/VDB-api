# Leetcode Solution Provider
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

MILVUS_INDEX_PARAM={"metric_type": "L2", "index_type": "HNSW", "params": {"M": 8, "efConstruction": 200}}

milvusdb = MilvusDB(index_param=MILVUS_INDEX_PARAM)


class LeetCodeSolutionFullProvider(BaseProvider):
    def __init__(self):
        super().__init__(collection_name="leetcode_solution_full", uid_field="solution_id")
        
    def get_schema(self):
        fields = [
            FieldSchema(name='solution_id', dtype=DataType.VARCHAR, max_length=6400, is_primary=True),
            FieldSchema(name='problem_id', dtype=DataType.INT64),
            FieldSchema(name='description', dtype=DataType.VARCHAR, max_length=64000),
            FieldSchema(name='solution', dtype=DataType.VARCHAR, max_length=64000),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        ]
        embed_field = 'embedding'

        return fields, embed_field

    def get_output_fields(self):
        return ["solution_id", "problem_id", "description", "solution"]

    def parse_data(self, json_data):
        collection = milvusdb.connect_collection(self.collection_name)
        existing_ids = get_existing_solution_ids(
            collection, 
            self.uid_field, 
            is_int=False)

        with tqdm(total=len(json_data)) as pbar:
            logger.info("Start parsing data...")
            for element in json_data:
                if element["solution_id"] in existing_ids:
                    pbar.set_description(f"Skipping {element['solution_id']}")
                    logger.info(f"Skipping {element['solution_id']}")
                    pbar.update(1)
                    continue

                pbar.set_description(f"Embedding {element['solution_id']}")
                logger.info(f"Embedding {element['solution_id']}")
                merged_content = element["description"] + element["solution"]
                cut_content = embedder.truncate_to_tokens(merged_content)  # 최대 토큰 길이로 자르기
                array_data = [
                    [element["solution_id"]],
                    [element["problem_id"]],
                    [element["description"]],
                    [element["solution"]],
                    embedder.embed(cut_content)
                ]
                milvusdb.ingest(collection, array_data)
                pbar.update(1)
        logger.info("Finish parsing data...")
    
__all__ = [
    "LeetCodeSolutionFullProvider"
]