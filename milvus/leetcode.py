import os
import json
import pandas as pd
from tqdm import tqdm
from pymilvus import FieldSchema, DataType
from milvus.milvus_router import MilvusDB
from util.token_cutter import truncate_to_tokens
from util.existing_checker import get_existing_solution_ids

DIMENSION = int(os.getenv("MILVUS_DIMENSION"))
milvusdb = MilvusDB()

leetcode_solution_fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='problem_id', dtype=DataType.INT64),
    FieldSchema(name='solution_id', dtype=DataType.VARCHAR, max_length=6400),
    FieldSchema(name='description', dtype=DataType.VARCHAR, max_length=64000),
    FieldSchema(name='solution', dtype=DataType.VARCHAR, max_length=64000),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
]

async def process_file(file, collection_name):
    file_type = file.content_type
    if file_type == 'application/json':
        file_content = await file.read()
        data = json.loads(file_content.decode('utf-8'))
    
    collection = milvusdb.connect_collection(collection_name)
    existing_solution_ids = get_existing_solution_ids(collection, "solution_id")  # 기존 데이터 조회
    
    with tqdm(total=len(data)) as pbar:
        for element in data:
            data_array = [
                [], # 0 problem_id
                [], # 1 solution_id
                [], # 2 description
                [] # 3 solution
            ]
            solution_id = element['solution_id']
            
            if solution_id in existing_solution_ids:
                pbar.set_description(f"Skipping {element['problem_id']} > {solution_id}")
                pbar.update(1)
                continue  # 중복 건너뛰기

            pbar.set_description(f"Inserting {element['problem_id']} --> {solution_id}")
            
            merged_content = element["description"] + element["solution"]
            cut_content = truncate_to_tokens(merged_content)  # 최대 토큰 길이로 자르기
            
            data_array[0].append(element['problem_id'])
            data_array[1].append(element['solution_id'])
            data_array[2].append(element['description'])
            data_array[3].append(element['solution'])
            data_array.append(milvusdb.embed(cut_content))  # content를 embedding
            
            milvusdb.ingest(collection=collection, data=data_array)  # 하나씩 삽입
            
            pbar.update(1)