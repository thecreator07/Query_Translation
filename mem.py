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
# queque system for memory
memory_queue = Queue()
def memory_writer():
    while True:
        batch = []
        while not memory_queue.empty():
            batch.append(memory_queue.get())

        if batch:
            for messages, user_id in batch:
                mem_client.add(messages, user_id=user_id)
        time.sleep(2)  # Flush every 2 seconds
        
threading.Thread(target=memory_writer, daemon=True).start()

gemini_client=OpenAI(
    api_key= os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

history_messages=[]
def chat(message):
    mem_result=mem_client.search(query=message,user_id='p82')
    
    messages=[
        {"role":"user","content":message}
    ]
        
    memories = "\n".join([m["memory"] for m in mem_result.get("results")])

    print(f"\n\nMEMORY:\n\n{memories}\n\n")
    
    SYSTEM_PROMPT = f"""
        You have to answer the question what user is asking.you have the world knowledge 
        
        user previous conversation:
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
    memory_queue.put((messages, 'p82'))  # enqueue instead of direct add
    return result.choices[0].message.content

while True:
    query=input("user: ")
    result=chat(query)
    print("Bot: ",result)