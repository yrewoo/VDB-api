import os
from tqdm import tqdm
from src.milvus_router import MilvusDB
from pymilvus import FieldSchema, DataType, utility
from src.util.token_cutter import truncate_to_tokens
from src.providers.base_provider import BaseProvider
from src.util.existing_checker import get_existing_solution_ids

DIMENSION = int(os.getenv("MILVUS_DIMENSION"))
milvusdb = MilvusDB()

class LeetCodeSolutionProvider(BaseProvider):
    def __init__(self):
        self.collection_name = "leetcode_solution"
        self.uid_field = "solution_id"

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

    def parse_data(self, json_data):
        collection = milvusdb.connect_collection(self.collection_name)
        existing_ids = get_existing_solution_ids(
            collection, 
            self.uid_field, 
            is_int=False)

        with tqdm(total=len(json_data)) as pbar:
            for element in json_data:
                if element["solution_id"] in existing_ids:
                    pbar.set_description(f"Skipping {element['solution_id']}")
                    pbar.update(1)
                    continue

                pbar.set_description(f"Embedding {element['solution_id']}")
                merged_content = element["description"] + element["solution"]
                cut_content = truncate_to_tokens(merged_content)  # 최대 토큰 길이로 자르기
                array_data = [
                    [element["solution_id"]],
                    [element["problem_id"]],
                    [element["description"]],
                    [element["solution"]],
                    milvusdb.embed(cut_content)
                ]
                pbar.update(1)
                milvusdb.ingest(collection, array_data)
    
__all__ = [
    "LeetCodeSolutionProvider"
]