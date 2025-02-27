# Grepp Solution Provider

from tqdm import tqdm
from src.config import DIMENSION
from src.util.logger import logger
from src.milvus_router import MilvusDB
from pymilvus import FieldSchema, DataType
from src.util.embedding_model import embedder
from src.providers.base_provider import BaseProvider
from src.util.existing_checker import get_existing_solution_ids

milvusdb = MilvusDB()

class GreppSolutionProvider(BaseProvider):
    def __init__(self):
        super().__init__(collection_name="grepp_solution", uid_field="solution_id")
        
    def get_schema(self):
        fields = [
            FieldSchema(name='solution_id', dtype=DataType.INT64, is_primary=True),
            FieldSchema(name='problem_id', dtype=DataType.INT64),
            FieldSchema(name='language', dtype=DataType.VARCHAR, max_length=6400),
            FieldSchema(name='description', dtype=DataType.VARCHAR, max_length=64000),
            FieldSchema(name='code', dtype=DataType.VARCHAR, max_length=64000),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        ]
        embed_field = 'embedding'
        return fields, embed_field

    def get_output_fields(self):
        return ["solution_id", "problem_id", "language", "description", "code"]

    def parse_data(self, json_data):
        collection = milvusdb.connect_collection(self.collection_name)
        data = json_data

        existing_ids = get_existing_solution_ids(
            collection, 
            self.uid_field, 
            is_int=True)

        with tqdm(total=len(data)) as pbar:
            logger.info("Start parsing data...")
            for element in data:
                solution_id = element["solution_id"]
                if solution_id in existing_ids:
                    pbar.set_description(f"Skipping {solution_id}")
                    logger.info(f"Skipping {solution_id}")
                    pbar.update(1)
                    continue

                pbar.set_description(f"Embedding {solution_id}")
                logger.info(f"Embedding {solution_id}")
                merged_content = element["description"] + element["code"]
                cut_content = embedder.truncate_to_tokens(merged_content)  # 최대 토큰 길이로 자르기
                array_data = [
                    [solution_id],              # solution_id
                    [element['problem_id']],    # problem_id
                    [element['language']],
                    [element['description']],   # problem's description
                    [element['code']],
                    embedder.embed(cut_content)
                ]
                milvusdb.ingest(collection, array_data)
                pbar.update(1)
        logger.info("Finish parsing data...")

        # collection = milvusdb.connect_collection(self.collection_name)
        # data = json_data['challenges']
        # existing_ids = get_existing_solution_ids(
        #     collection, 
        #     self.uid_field, 
        #     is_int=True)

        # total_size = sum(len(element.get("solutionGroups", [])) for element in data)
        # with tqdm(total=total_size) as pbar:
        #     for element in data:
        #         description = element['description']
        #         for solution in element['solutionGroups']:
        #             solution_id = solution['id']
        #             if solution_id in existing_ids:
        #                 pbar.set_description(f"Skipping {solution_id}")
        #                 pbar.update(1)
        #                 continue

        #             pbar.set_description(f"Embedding {solution_id}")
        #             merged_content = description + solution['code']
        #             cut_content = embedder.truncate_to_tokens(merged_content)  # 최대 토큰 길이로 자르기
        #             array_data = [
        #                 [solution['id']],           # solution_id
        #                 [solution['challengeId']],  # problem_id
        #                 [solution['language']],
        #                 [solution['code']],
        #                 embedder.embed(cut_content)
        #             ]
        #             milvusdb.ingest(collection, array_data)
        #             pbar.update(1)
    
__all__ = [
    "GreppSolutionProvider"
]