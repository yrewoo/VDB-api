import os
import json

from pymilvus import (
    connections,
    utility,
    CollectionSchema,
    Collection,
)
from openai import OpenAI
from dotenv import load_dotenv
from src.util.logger import logger
from fastapi.encoders import jsonable_encoder


class MilvusDB():
    def __init__(self, host=None, port=None, embedding_model=None, api_key=None):
        load_dotenv(override=True)

        self.host = host or os.getenv("MILVUS_HOST", "localhost")
        self.port = port or int(os.getenv("MILVUS_PORT", 19530))
        self.index_param = json.loads(os.getenv("MILVUS_INDEX_PARAM"))
        self.query_param = json.loads(os.getenv("MILVUS_QUERY_PARAM"))
        self.embedding_model = embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

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
            collection.create_index(field_name=f"{embed_field}", index_params=self.index_param)
            collection.load()
            print(f"============= <Collection: {collection_name}> Created")
            return collection
        except Exception as e:
            logger.error(e)
            raise e
    
    def embed(self, texts):
        try:
            client = OpenAI(api_key = self.api_key)
            embeddings = client.embeddings.create(
                input=texts,
                model=self.embedding_model
            )
            return [x.embedding for x in embeddings.data]
        except Exception as e:
            logger.error(e)
            raise e
    
    def ingest(self, collection, data):
        try:
            collection.insert(data=data)
        except Exception as e:
            logger.error(e)
            raise e

    def query(self, collection, fields, expr, limit=1):
        try:
            output_fields = fields
            result = collection.query(
                expr = expr,
                output_fields = output_fields,
                limit = limit,
            )
            return result
        except Exception as e:
            logger.error(e)
            raise e

    def search(self, collection, data, target, output_fields, top_k=5, expr=None):
        try:
            embeddings = self.embed(data)
            outputs = collection.search(
                data=embeddings, 
                anns_field=target, 
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
