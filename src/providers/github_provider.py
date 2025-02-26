# Github Provider

from tqdm import tqdm
from src.config import DIMENSION
from src.milvus_router import MilvusDB
from pymilvus import FieldSchema, DataType
from src.util.embedding_model import embedder
from src.providers.base_provider import BaseProvider
from src.util.existing_checker import get_existing_solution_ids

milvusdb = MilvusDB()


class GithubProvider(BaseProvider):
    def __init__(self):
        super().__init__(collection_name="github", uid_field="file_name")
        
    def get_schema(self):
        fields = [
            FieldSchema(name='file_name', dtype=DataType.VARCHAR, max_length=10000, is_primary=True),
            FieldSchema(name='line_count', dtype=DataType.INT64),
            FieldSchema(name='mark', dtype=DataType.FLOAT),
            FieldSchema(name='code', dtype=DataType.VARCHAR, max_length=64000),
            FieldSchema(name='query', dtype=DataType.VARCHAR, max_length=64000),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        ]
        embed_field = 'embedding'
        return fields, embed_field

    def get_output_fields(self):
        return ["file_name", "line_count", "mark", "code", "query"]
    
    def parse_data(self, json_data):
        collection = milvusdb.connect_collection(self.collection_name)
        existing_ids = get_existing_solution_ids(
            collection, 
            self.uid_field, 
            is_int=False)
        
        with tqdm(total=len(json_data)) as pbar:
            for element in json_data:
                if element["file_name"] in existing_ids:
                    pbar.set_description(f"Skipping {element['file_name']}")
                    pbar.update(1)
                    continue

                pbar.set_description(f"Embedding {element['file_name']}")
                merged_content = element["code"] + element["query"]
                cut_content = embedder.truncate_to_tokens(merged_content)  # 최대 토큰 길이로 자르기
                array_data = [
                    [element["file_name"]],
                    [element["line_count"]],
                    [element["mark"]],
                    [element["code"]],
                    [element["query"]],
                    embedder.embed(cut_content)
                ]
                pbar.update(1)
                milvusdb.ingest(collection, array_data)
    
__all__ = [
    "GithubProvider"
]
