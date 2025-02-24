# For Creating Milvus Collections
import argparse
from util.logger import logger
from milvus.milvus_router import MilvusDB
from milvus.leetcode import leetcode_solution_fields
from milvus.grepp import grepp_fields, grepp_solution_fields 
from milvus.custom import fields

def initialize(collection_name, fields, embed_field='embedding'):
    milvusdb = MilvusDB()
    collection = milvusdb.create_collection(
        collection_name=collection_name, 
        fields=fields, 
        embed_field=embed_field
    )

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--collection', required=True)
    args = argparse.parse_args()

    if args.collection == "grepp":
        initialize(args.collection, grepp_fields)
    if args.collection == "grepp_solution":
        initialize(args.collection, grepp_solution_fields)
    if args.collection == "leetcode_solution":
        initialize(args.collection, leetcode_solution_fields)
    if args.collection == "crawling":
        initialize(args.collection, fields)
    # Add more collections here ...
        # ...
