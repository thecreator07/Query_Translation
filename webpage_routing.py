from webpage_loader import get_internal_links, docs_splitter
from urllib.parse import urlparse
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
# from openai import OpenAI

# getting all the links from given url
urls=get_internal_links("https://docs.chaicode.com/youtube/getting-started/")
valid_urls = [url for url in urls if url.startswith("https://docs.chaicode.com/youtube/chai")]

sorted_url=sorted(valid_urls)
print("\n".join([url for url in sorted_url]))

collections=set() #unique 

for url in sorted_url:
    path_parts = urlparse(url).path.split('/')
    collection_name=path_parts[2]
    collections.add(collection_name)

# google embedder - embedding-001  
embedder=GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.environ.get("GEMINI_API_KEY"))

client = QdrantClient(url="http://localhost:6333")
collection_of_Db = client.get_collections().collections   
existing = {c.name for c in collection_of_Db}

collections=collections-existing   #if new pages added in future
print("all collections:",collections)

if len(collections) > 0:
    # Inserting docs in QdrantDb
    for collection in collections:
        pages_url = [u for u in valid_urls if f"/{collection}/" in u]
        print(f"{collection} db start ingestion")

        for page in pages_url:
            split_docs = docs_splitter(page)
            # print(f"page indexed:\n {split_docs}")
            vector_store = QdrantVectorStore.from_documents(
                documents=split_docs,
                url="http://localhost:6333",
                collection_name=collection,
                embedding=embedder
            )
            print(f"{page} is indexed")

        print(f"{collection} DB created and populated 🗂️")
else:
    print("✅ No new collections to ingest, skipping loop.")   





    
# vector_store.add_documents(split_docs)
# print("Injection Done")    
    


# print(docs[0])