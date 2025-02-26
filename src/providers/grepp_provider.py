import os
from tqdm import tqdm

from src.config import DIMENSION
from src.milvus_router import MilvusDB
from pymilvus import FieldSchema, DataType
from src.util.embedding_model import embedder
from src.providers.base_provider import BaseProvider
from src.util.existing_checker import get_existing_solution_ids

milvusdb = MilvusDB()

class GreppProvider(BaseProvider):
    def __init__(self):
        super().__init__(collection_name="grepp", uid_field="problem_id")

    def get_schema(self):
        fields = [
            FieldSchema(name='problem_id', dtype=DataType.INT64, is_primary=True),
            FieldSchema(name='title', dtype=DataType.VARCHAR, max_length=64000),
            FieldSchema(name='partTitle', dtype=DataType.VARCHAR, max_length=64000),
            FieldSchema(name='languages', dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name='level', dtype=DataType.INT64),
            FieldSchema(name='description', dtype=DataType.VARCHAR, max_length=64000),
            FieldSchema(name='testcases', dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        ]
        embed_field = 'embedding'
        return fields, embed_field

    def parse_data(self, json_data):
        collection = milvusdb.connect_collection(self.collection_name)
        data = json_data['challenges']
        existing_ids = get_existing_solution_ids(
            collection, 
            self.uid_field, 
            is_int=True)

        with tqdm(total=len(data)) as pbar:
            for element in data:
                problem_id = element['id']
                if problem_id in existing_ids:
                    pbar.set_description(f"Skipping {problem_id}")
                    pbar.update(1)
                    continue

                pbar.set_description(f"Embedding {problem_id}")
                array_data = [
                    [element['id']],                        # problem_id
                    [element['title']],                     # title
                    [element['partTitle']],                 # partTitle
                    [str(element['languages'])],            # languages
                    [element['level']],                     # level
                    [element['description']],               # description
                    [str(element['testcases'])],            # testcases
                    embedder.embed(element["description"])  # embedding
                ]
                milvusdb.ingest(collection, array_data)
                pbar.update(1)
    
__all__ = [
    "GreppProvider"
]