from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


client = QdrantClient(url="http://localhost:")
collection_of_Db = client.get_collections().collections
existing = {c.name for c in collection_of_Db}
 
print(existing)
for collection in existing:
    client.delete_collection(collection)
    
