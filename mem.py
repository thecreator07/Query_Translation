from mem0 import Memory
import os
from openai import OpenAI

config = {
    "version": "v1.1",
    "embedder": {
        "provider": "gemini",
        "config": {
            "api_key": os.environ.get("GEMINI_API_KEY"),
            "model": "models/text-embedding-004",
            "embedding_dims": 768  # ← Must match actual embedding output
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

gemini_client=OpenAI(
    api_key= os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


def chat(message):
    mem_result=mem_client.search(query=message,user_id="p81")
    
    messages=[
        {"role":"user","content":message}
    ]
        
    memories = "\n".join([m["memory"] for m in mem_result.get("results")])

    print(f"\n\nMEMORY:\n\n{memories}\n\n")
    
    SYSTEM_PROMPT = f"""
        You are memory aware Ai assistant who have the world knowledge
        Tone: Professional analytical, precision-focused, with clear uncertainty signaling
        
        Memory
        {memories}
    """
    
    messages = [
        { "role": "system", "content": SYSTEM_PROMPT },
        { "role": "user", "content": message }
    ]
    
    result=gemini_client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=messages  
    )
    
    messages.append({"role":"assistant","content":result.choices[0].message.content})
    mem_client.add(messages,"p81")
    return result.choices[0].message.content

while True:
    query=input("user: ")
    result=chat(query)
    print("Bot: ",result)