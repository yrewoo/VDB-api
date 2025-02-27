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

# 1. Start Database Contrainer
```shell
$ cd docker
$ docker-compose -f milvus.yml up -d
```
# 2. Define the Collection Provider
Before starting the API server, `{collection_name}_provider.py` under `./src/providers/` should be created.

# 3. Start the API server
```shell
$ uvicorn src.main:app --host $HOST --port $PORT
```
You can use Swagger UI on `API_endpoint/docs` (e.g. `http://$HOST:$PORT/docs`).

## 3-1. Upload data (`/upload`)
The Upload API requires two input parameters: `collection_name` and `file`. Uploading files is handled as a background task. To check the status of the job in the background after running the upload API, call the `/upload_status/` API.

## 3-2. Data Search (`/expr_search`)
### Example 1 - Using cURL
To retrieve data with problem_id 16769 from The Collection “grepp”, `expr` would be `problem_id==16769` as following:
```shell
curl -X 'GET' \
    "http://0.0.0.0:10001/expr_search/?collection_name=grepp&expr=problem_id==16769&limit=1" \
    -H "accept: application/json"
```
The result is, 
```json
{
    "total": 1,
    "result": [
        {
            "problem_id": 16769,
            "title": "빠른 이동",
            "partTitle": "2023 현대모비스 알고리즘 경진대회 본선",
            "languages": "['c', 'cpp', ...]",
            "level": 5,
            "description": "...",
            "testcases": "[{'input':'...', 'output':'...'}, ...]"
        }
    ]
}
```
### Example 2 - Using Python `requests`
To retrieve three data(`limit=3`) with "출력" in the title from The Collection “grepp”, `expr` would be `title LIKE "%출력%"` as following (when using cURL, space and % should be encoded as `%20` and `%25`):
```python
import requests

url = "http://0.0.0.0:10001/expr_search/"
params = {
    "collection_name": "grepp",
    "expr": "title LIKE '%출력%'",
    "limit": 3
}

response = requests.get(url, params=params)
```
The result is, 
```json
{
    "total": 3,
    "result": [
        {
            "problem_id": 17582,
            "title": "문자열 출력하기",
            ...
        },
        {
            ...
        },
        {
            ...
        }
    ]
}
```

## 3-3. Vector Search (`/vector_search`)
### Example
To vector search for problem descriptions similar to “Write a function argumentsLength that returns the count of arguments passed to it.”, 
```python
import requests

url = "http://0.0.0.0:10001/vector_search/"
params = {
    "collection_name": "grepp",
    "text": “Write a function argumentsLength that returns the count of arguments passed to it.”,
    "limit": 3
}

response = requests.get(url, params=params)
```
The result is,
```json
{
    "total": 3,
    "result": [
        {
            "id": "18196",
            "distance": 1.4860655069351196,
            "entity": {
                "problem_id": 18196,
                "title": "상담원 인원",
                ...
            }
        },
        {
            "id": "17735",
            "distance": 1.4864739179611206,
            "entity": {
                ...
            }
        },
        {
            "id": "17653",
            "distance": 1.4897608757019043,
            "entity": {
                ...
            }
        }
    ]
}
```