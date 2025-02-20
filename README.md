# VDB API
## Preparation
1. Make virtual environment and activate virtual env. 
```shell
$ virtualenv .venv -p python3.10
$ source .venv/bin/activate
```
2. Install packages.
```shell
$ pip install -r requirements.txt
```
3. Set `.env` file

Set up Milvus connection config(e.g. host, port, ...) and OpenAI's API key (required for the embedding model).

## 1. Start Database Contrainer
```shell
$ cd docker
$ docker-compose -f milvus.yml up -d
```
## 2. Initialize Database
Before running, make sure that the Collection's Fields are defined in the form of `milvus/collection_name.py`. If necessary, you should add an initialization step for creating the collection in the `milvus/init.py` file. You can see DB GUI using `attu` (check endpoint at `docker/milvus.yml`)
```shell
$ python milvus/init.py --collection "collection_name"
```
### 3. Start Fast API
```shell
$ python main.py
```
You can use Swagger UI on `API_endpoint/docs` (e.g. `http://localhost:port/docs`).
### 3-1. Upload data (`/upload`)
The Upload API requires two input parameters: `collection_name` and `file`. 

**Before uploading**
1. The collection must be created at vector database using initializing(step #2).
2. The `process_file` function in `milvus/collection_name.py` must be defined to match the data structure of the JSON file.
3. The `insert_data` function in `milvus/collection_router.py` must be defined to match the collection.

### 3-2. Data Search (`/search_expr`)
TBD

### 3-3. Vector Search (`/vector_search`)
TBD