from pathlib import Path
from annotated_types import doc
from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from helper import reciprocal_rank_fusion


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
you are helpful assistant that generates 5 duplicate questions based on the given question:{search_query}

rules:
- every generate question should be differentiate using "\\\n"

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

# store ranked chunks
ranked_docs=[]
for question in (parallel_questions.choices[0].message.content).split("\\\n"):
    relevant_chunks = retriver.similarity_search(question)
    relevant_chunks=[chunk.page_content for chunk in relevant_chunks]

    # print(f"Relevant Chunks:", relevant_chunks)
    ranked_docs.append(relevant_chunks)

fused_docs = reciprocal_rank_fusion(ranked_docs,k=60.0)
    
# join all unique chunks into a single context
final_context = ",".join(doc for doc, _ in fused_docs[:5])
print("Final Context:", final_context)
final_LLM_SYSTEM_PROMPT ="""You are a helpful assistant that gives elaborate answers to questions based on the available context:{final_context}

rules:
1. don't include the 'context' word in your answer.
2. if code then provide the code in markdown format.
3. format your answer in a way that is easy to read and understand.
"""

final_query_Answer=client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {"role": "system", "content": final_LLM_SYSTEM_PROMPT.format(final_context=final_context)},
        {"role": "user", "content": search_query}
    ]
)


print("\nFinal Answer:", final_query_Answer.choices[0].message.content)