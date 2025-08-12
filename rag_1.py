from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "text-to-speech-467917-67fb31ad2152.json"

pdf_path=Path(__file__).parent /"nodejs.pdf"
loader=PyPDFLoader(file_path=pdf_path)

docs=loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs=text_splitter.split_documents(docs)


embedder=GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.environ.get("GEMINI_API_KEY"))


vector_store=QdrantVectorStore.from_documents(
    documents=[], # Start with an empty collection
    url="http://localhost:6333",
    collection_name="rag_1",
    embedding=embedder
)
    
vector_store.add_documents(split_docs)
print("Injection Done")
# AstraDb

retriver=QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="rag_1",
    embedding=embedder
)

search_query=input("Enter your question: ")

relevant_chunks=retriver.similarity_search(search_query)

print("relevant_chunks:", relevant_chunks)
relevant_chunks = "\n".join([chunk.page_content for chunk in relevant_chunks])

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the available context:{relevant_chunks}

rules:
1. answer the question based on the context provided.
2. don't include the 'context' word in your answer.
3. if code then provide the code in markdown format.
"""

client=OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
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