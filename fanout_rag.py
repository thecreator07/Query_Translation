from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

pdf_path=Path(__file__).parent /"nodejs.pdf"
loader=PyPDFLoader(file_path=pdf_path)

docs=loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs=text_splitter.split_documents(docs)

embedder=GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.environ.get("GEMINI_API_KEY"))

retriver=QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="rag_1",
    embedding=embedder
)

client=OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


search_query=input("Enter your question: ")
FanOut_SYSTEM_PROMPT="""
you are Ai assistant. you have to generate 3 similar questions based on the question:{search_query}

rules:
- every generate question should be differentiate using "\\\n"
- the questions should be related to the original question.
- generate 3 similar questions based on the question.
- don't include the 'question' word in your answer.
"""
parallel_questions=client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {"role": "system", "content": FanOut_SYSTEM_PROMPT},
        {"role": "user", "content": search_query}
    ]
)

print("Parallel Questions:", parallel_questions.choices[0].message.content)
print((parallel_questions.choices[0].message.content).split("\\\n"))

# store unique chunks
unique_chunks=[]
for question in (parallel_questions.choices[0].message.content).split("\\\n"):
    relevant_chunks = retriver.similarity_search(question)
    # filter out duplicates(only unique chunks are stored)
    for doc in relevant_chunks:
        if doc.page_content not in unique_chunks:
            unique_chunks.append(doc.page_content)

# join all unique chunks into a single context
final_context = ",".join(unique_chunks)
print("Final Context:", final_context)
final_LLM_SYSTEM_PROMPT ="""You are a helpful assistant that gives elaborate answers to questions based on the available context:{final_context}

rules:
1. answer the question based on the context provided.
2. don't include the 'context' word in your answer.
3. if code then provide the code in markdown format.
"""

final_query_Answer=client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {"role": "system", "content": final_LLM_SYSTEM_PROMPT.format(final_context=final_context)},
        {"role": "user", "content": search_query}
    ]
)


print("\nFinal Answer:", final_query_Answer.choices[0].message.content)