from mem0 import Memory
import os
from openai import OpenAI
from queue import Queue
import threading
import time

config = {
    "version": "v1.1",
    "embedder": {
        "provider": "gemini",
        "config": {
            "api_key": os.environ.get("GEMINI_API_KEY"),
            "model": "models/text-embedding-004",
            "embedding_dims": 768  
        },
    },
    "vector_store": {
    "provider": "qdrant",
    "config": {
        "host": "localhost",
        "port": 6333,
        "embedding_model_dims": 768
    },
},
    "llm": {
        "provider": "gemini",
        "config": {
            "api_key": os.environ.get("GEMINI_API_KEY"),
            "model": "gemini-2.0-flash-001",
            "temperature": 0.2
        },
    },    
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": os.environ.get("NEO4J_URL"),
            "username": os.environ.get("NEO4J_USERNAME"),
            "password": os.environ.get("NEO4J_PASSWORD")
        }
    }
}


mem_client=Memory.from_config(config)

messages = [
        { "role": "system", "content": "you are a good ai assistant" },
        { "role": "user", "content": "i want to go my fabourite place" }
]

result = mem_client.search("what does i like?", user_id='p82')
# result.facts => ["Name is Aman Kumar.", "Graduated from HNBGU, Srinagar, Uttarakhand"]
# facts = result
    
print("\n".join([f'Memory: {res["memory"]}, Score: {res["score"]}' for res in result["results"]]))
