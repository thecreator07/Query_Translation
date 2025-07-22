from webpage_loader import get_internal_links, docs_splitter
from urllib.parse import urlparse
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from openai import OpenAI

client = QdrantClient(url="http://localhost:6333")
collection_of_Db = client.get_collections().collections   
existing = {c.name for c in collection_of_Db}


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


newcollections=collections-existing   #if new pages added in future
print("all collections:",collections)
# Ingestion
if len(newcollections) > 0:
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

        print(f"{collection} DB created and populated ")
else:
    print("✅ No new collections to ingest, skipping loop.")   

# Injection of website data completed


user_query=input("Ask The Question: ")



# Retrieving
#Routing to find collection 
SYSTEM_PROMPT1=f"""You are a vectorDb collection routing system.
based on the user_query:{user_query}, you have to choose which collection is best suited from available collections. 

Available collections:
{"\n".join([collection for collection in existing])}

Example:
Query:what is the nested if else?
Output: chai-aur-c
"""
print(SYSTEM_PROMPT1)
client=OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
collection_finding=client.chat.completions.create( model="gemini-2.0-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT1},
        {"role": "user", "content": user_query}
    ])

print(collection_finding.choices[0].message.content)
exact_collection=collection_finding.choices[0].message.content



# Retrieving of chunk
retriver=QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name=exact_collection,
    embedding=embedder
)

relevant_chunk=retriver.similarity_search(user_query)

print(relevant_chunk)
# Retriever of chunk data completed




# Generation
relevant_context=[{"content": chunk.page_content, "source": chunk.metadata["source"]} for chunk in relevant_chunk]
SYSTEM_PROMPT = f"""
you are the the teacher which role is to provide answer based on context:{"\n".join([chunk["content"] for chunk in relevant_context])} with all sources:{"\n".join([chunk["source"] for chunk in relevant_context])}, which can be useful to the user's question: {user_query}

Rules:
    - give Output based on provided context
    
Example:
Input:what is nginx
Output:Nginx is a web server that can be used to serve static files. The documents also describe how to install, configure, and use Nginx for tasks like deploying Node APIs and implementing rate limiting. 
       .................
       sources:
              url1,
              url2
                
"""
print(SYSTEM_PROMPT)
chat=client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
)

print("Answer:", chat.choices[0].message.content)
