import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_internal_links(base_url):
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = set()
    for a in soup.find_all('a', href=True):
        href = a['href']
        # Only keep internal links that start with the base path
        
        links.add("https://docs.chaicode.com" + href)
    return list(links)

def docs_splitter(base_url):
    loader = WebBaseLoader(base_url)
    docs=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs=text_splitter.split_documents(docs)
    return split_docs