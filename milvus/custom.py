import os
from pymilvus import FieldSchema, DataType

DIMENSION = int(os.getenv("MILVUS_DIMENSION"))

fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True),
    # ...
]

async def process_file(file, collection_name):
    # ...
    pass