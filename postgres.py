from webpage_loader import get_internal_links, docs_splitter
from urllib.parse import urlparse
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from openai import OpenAI
from langchain_postgres import PGVector
from langchain_community.document_loaders import WebBaseLoader
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "text-to-speech-467917-67fb31ad2152.json"

responce=requests.get('https://nodejs.org/docs/latest/api/')

soup=BeautifulSoup(responce.text,'html.parser')

links=set()
for a in soup.find_all('a', href=True):
    href = a['href']
    links.add(href)

# print("\n".join([link for link in links]))

valid_urls = [url for url in links if url.startswith("https://nodejs.org/docs/latest/api/")]

print("\n".join([url for url in valid_urls]))

def docs_splitter(base_url):
    loader = WebBaseLoader(base_url)
    docs=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs=text_splitter.split_documents(docs)
    return split_docs

# split_docs=docs_splitter('https://docs.astral.sh/uv/getting-started/installation/')

embedder=GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.environ.get("GEMINI_API_KEY"))


vector_store = PGVector(
    embeddings=embedder,
    collection_name="docsUV",
    connection='postgresql://postgres:Aman%40776281%40%23@db.ognsvaxwyiucxgscnutf.supabase.co:5432/postgres',
    use_jsonb=True,
    create_extension=True  # since it's already enabled
)

# print(split_docs)
# vector_store.add_documents(documents=split_docs)

search_query=input("Enter Your Question ")

relevant_chunk=vector_store.similarity_search(search_query)

print(relevant_chunk)


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the available context:{relevant_chunks}

rules:
1. answer the question based on the context provided.
2. don't include the 'context' word in your answer.
3. if code then provide the code in markdown format.
"""


client=OpenAI(
    api_key='AIzaSyBwhFgQTUI9Vw--rnhrjzKs0z4kRkHMPyw',
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

chat=client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": search_query}
    ]
)

print("Answer:", chat.choices[0].message.content)