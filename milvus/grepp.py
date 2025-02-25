import os
import json
from tqdm.asyncio import tqdm
from milvus.milvus_router import MilvusDB
from util.existing_checker import get_existing_solution_ids
from util.token_cutter import truncate_to_tokens
from pymilvus import FieldSchema, DataType

DIMENSION = int(os.getenv("MILVUS_DIMENSION"))
milvusdb = MilvusDB()

grepp_fields = [
    FieldSchema(name='problem_id', dtype=DataType.INT64, is_primary=True),
    FieldSchema(name='title', dtype=DataType.VARCHAR, max_length=64000),
    FieldSchema(name='partTitle', dtype=DataType.VARCHAR, max_length=64000),
    FieldSchema(name='languages', dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name='level', dtype=DataType.INT64),
    FieldSchema(name='description', dtype=DataType.VARCHAR, max_length=64000),
    FieldSchema(name='testcases', dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
]

grepp_solution_fields = [
    FieldSchema(name='solution_id', dtype=DataType.INT64, is_primary=True),
    FieldSchema(name='problem_id', dtype=DataType.INT64),
    FieldSchema(name='language', dtype=DataType.VARCHAR, max_length=6400),
    FieldSchema(name='code', dtype=DataType.VARCHAR, max_length=64000),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
]

def prepare_data_array(collection_name, element, solution=None):
    if collection_name == "grepp":
        return [
            [element['id']],  # problem_id
            [element['title']], 
            [element['partTitle']],
            [str(element['languages'])],
            [element['level']],
            [element['description']],
            [str(element['testcases'])],
        ]
    
    elif collection_name == "grepp_solution":
        return [
            [solution['id']],  # solution_id
            [solution['challengeId']],  # problem_id
            [solution['language']],
            [solution['code']],
        ]

# ✅ Milvus에 데이터를 삽입하는 공통 함수
def insert_data_to_milvus(collection, data_array, text_for_embedding):
    cut_content = truncate_to_tokens(text_for_embedding)  # 최대 토큰 길이로 자르기
    data_array.append(milvusdb.embed(cut_content))  # 임베딩 추가
    milvusdb.ingest(collection, data_array)  # Milvus에 삽입

# ✅ 컬렉션을 처리하는 공통 함수
async def process_file(file, collection_name):
    file_content = await file.read()
    json_data = json.loads(file_content.decode('utf-8'))
    data = json_data['challenges']

    # 컬렉션 연결 및 기존 데이터 조회
    collection = milvusdb.connect_collection(collection_name)
    check_field = "solution_id" if collection_name == "grepp_solution" else "problem_id"
    existing_ids = get_existing_solution_ids(collection, check_field, is_int=True)

    total_size = sum(len(element.get("solutionGroups", [])) if collection_name == "grepp_solution" else 1 for element in data)

    with tqdm(total=total_size) as pbar:
        for element in data:
            if collection_name == "grepp":
                problem_id = element['id']
                if problem_id in existing_ids:
                    pbar.set_description(f"Skipping {problem_id}")
                    pbar.update(1)
                    continue

                pbar.set_description(f"Inserting {problem_id}")
                data_array = prepare_data_array(collection_name, element)
                insert_data_to_milvus(collection, data_array, element["description"])
                pbar.update(1)

            elif collection_name == "grepp_solution":
                description = element['description']
                for solution in element['solutionGroups']:
                    solution_id = solution['id']
                    if solution_id in existing_ids:
                        pbar.set_description(f"Skipping {solution_id}")
                        pbar.update(1)
                        continue

                    pbar.set_description(f"Inserting {solution_id}")
                    data_array = prepare_data_array(collection_name, element, solution)
                    merged_content = description + solution['code']
                    insert_data_to_milvus(collection, data_array, merged_content)

                    pbar.update(1)
