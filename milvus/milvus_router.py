import os

from pymilvus import (
    connections,
    utility,
    CollectionSchema,
    Collection,
)
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("MILVUS_HOST", "localhost")
PORT = os.getenv("MILVUS_PORT", 19530)
INDEX_PARAM = os.getenv("MILVUS_INDEX_PARAM")
QUERY_PARAM = os.getenv("MILVUS_QUERY_PARAM")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
API_KEY = os.getenv("OPENAI_API_KEY")

class MilvusDB():
    def __init__(self):
        pass

    def list_collections(self):
        connections.connect(host=HOST, port=PORT)
        return utility.list_collections()
    
    def connect_collection(self, collection_name):
        print(f"<Collection>:\n =============\n <Host:Port> {HOST}:{PORT}")
        connections.connect(host=HOST, port=PORT)
        collection = Collection(f"{collection_name}")      # Get an existing collection.
        collection.load()
        return collection
    
    def create_collection(self, collection_name, fields, embed_field, drop_existing=False):
        connections.connect(host=HOST, port=PORT)
        print(f"<Collection>:\n =============\n <Host:Port> {HOST}:{PORT}")
        print(f'<Name> {collection_name}')
        
        if drop_existing and utility.has_collection(f'{collection_name}'):
            utility.drop_collection(f'{collection_name}')
        
        schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
        collection = Collection(name=f'{collection_name}', schema=schema)
        collection.create_index(field_name=f"{embed_field}", index_params=INDEX_PARAM)
        collection.load()
        print(f"<Collection: {collection_name}> Created")
        return collection
    
    def embed(self, texts):
        client = OpenAI(api_key = API_KEY)
        embeddings = client.embeddings.create(
            input=texts,
            model=EMBEDDING_MODEL
        )
        return [x.embedding for x in embeddings.data]
    
    def ingest(self, collection, data):
        collection.insert(data=data)

    def query(self, collection, fields, expr, limit=1):
        output_fields = fields
        result = collection.query(
            expr = expr,
            output_fields = output_fields,
            limit = limit,
        )
        return result

    def search(self, collection, data, target, expr=None, top_k=5, output_fields=['problem_id', 'title', 'level',  'description', 'examples', 'constraints']):
        result = {}
        outputs = collection.search(
            data=self.embed(data), 
            anns_field=target, 
            expr=expr,
            param=QUERY_PARAM,
            limit=top_k,
            output_fields= output_fields,
          )
        response = []
        for output in outputs:
            for record in output:
                tmp = {
                    "id": record.id,
                    "distance": record.distance,
                    "entity": {}
                }
                for field in output_fields:
                    tmp["entity"].update({
                        f"{field}": record.get(field)
                    })
                response.append(tmp)
    
        result = {
            "response": sorted(response, key=lambda x: x['distance'])
        }

        return result