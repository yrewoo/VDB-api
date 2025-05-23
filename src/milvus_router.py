import os
import json
import numpy as np

from pymilvus import (
    connections,
    utility,
    CollectionSchema,
    Collection,
)
from dotenv import load_dotenv
from src.util.logger import logger
from src.util.embedding_model import embedder
from fastapi.encoders import jsonable_encoder

class MilvusDB():
    def __init__(self, host=None, port=None, index_param=None, query_param=None):
        # load_dotenv(override=True)

        self.host = host or os.getenv("MILVUS_HOST", "localhost")
        self.port = port or int(os.getenv("MILVUS_PORT", 19530))
        self.index_param = index_param or json.loads(os.getenv("MILVUS_INDEX_PARAM"))
        self.query_param = query_param or json.loads(os.getenv("MILVUS_QUERY_PARAM"))

    def list_collections(self):
        try:
            connections.connect(host=self.host, port=self.port)
            return utility.list_collections()
        except Exception as e:
            logger.error(e)
    
    def connect_collection(self, collection_name):
        try:
            connections.connect(host=self.host, port=self.port)
            collection = Collection(f"{collection_name}")      # Get an existing collection.
            collection.load()
            print(f"============= <Collection: {collection_name}> Connected")
            return collection
        except Exception as e:
            logger.error(e)
            raise e
    
    def create_collection(self, collection_name, fields, embed_field, drop_existing=False):
        try:
            connections.connect(host=self.host, port=self.port)
            print(f"============= <Host:Port> {self.host}:{self.port}")
            
            if drop_existing and utility.has_collection(f'{collection_name}'):
                utility.drop_collection(f'{collection_name}')
            
            schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
            collection = Collection(name=f'{collection_name}', schema=schema)
            if isinstance(embed_field, list):
                for field in embed_field:
                    collection.create_index(field_name=field, 
                                            index_name=f"{field}_index",
                                            index_params=self.index_param
                                        )
            else:
                collection.create_index(field_name=embed_field, index_params=self.index_param)
            
            collection.load()
            print(f"============= <Collection: {collection_name}> Created")
            
            if isinstance(embed_field, list):
                for field in embed_field:
                    print(collection.index(field_name=f"{field}", index_name=f"{field}_index").params)
            return collection
        except Exception as e:
            logger.error(e)
            raise e
    
    def ingest(self, collection, data):
        try:
            collection.insert(data=data)
        except Exception as e:
            logger.error(e)
            raise e


    def query(self, collection_name, output_fields, expr, limit=1, offset=0):
        try:
            collection = self.connect_collection(collection_name=collection_name)
            result = collection.query(
                expr=expr,
                output_fields=output_fields,
                limit=limit,
                offset=offset
            )

            # ✅ numpy.float32 → float 변환 함수
            def convert_values(obj):
                if isinstance(obj, np.float32):
                    return float(obj)  # ✅ 변환
                if isinstance(obj, list):
                    return [convert_values(item) for item in obj]  # ✅ 리스트 내부 변환
                if isinstance(obj, dict):
                    return {key: convert_values(value) for key, value in obj.items()}  # ✅ 딕셔너리 내부 변환
                return obj  # 기본 반환

            return convert_values(result)  # ✅ 변환 적용

        except Exception as e:
            logger.error(e)
            raise e

    def search(self, collection_name, data, target_field, output_fields, top_k=5, expr=None):
        try:
            collection = self.connect_collection(collection_name=collection_name)
            embeddings = embedder.embed(data)
            outputs = collection.search(
                data=embeddings, 
                anns_field=target_field, 
                expr=expr,
                param=self.query_param,
                limit=top_k,
                output_fields=output_fields,
            )
            response = []
            for hits in outputs:
                for hit in hits:
                    tmp = {
                        "id": hit.id if type(hit.id) == str else str(hit.id),
                        "distance": hit.distance,
                        "entity": {}
                    }
                    for field in output_fields:
                        tmp["entity"].update({
                            f"{field}": jsonable_encoder(hit.get(field))
                        })
                    response.append(tmp)
        
            response = sorted(response, key=lambda x: x['distance'])
            return response
        except Exception as e:
            logger.error(e)
            raise e
