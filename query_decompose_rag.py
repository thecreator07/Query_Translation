# from uuid import uuid4
import json
from pathlib import Path
from annotated_types import doc
from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
# from helper import reciprocal_rank_fusion


pdf_path=Path(__file__).parent /"pservicedec.pdf"
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

# ids = [str(uuid4()) for _ in split_docs]
# added_ids = retriver.add_documents(split_docs, ids=ids)
client=OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

search_query=input("Enter your question: ")
FanOut_SYSTEM_PROMPT=f"""
You are a helpful assistant that prepares queries that will be sent to a search component.
Sometimes, these queries are very complex.
Your job is to simplify complex queries into multiple queries that can be answered
in isolation to each-other.

Rules:
    - Follow the Output JSON Format.
    - if simple question, then Do not include the original question in the output- replicate 3 new questions.
    - if complex question, then replicate as Examples below.
    - Do Not Repeat the original question in the output.

Output Json format:
{{
    [{{question: "string", answer: "string"}}]
}}

Examples:
1. Query: Did Microsoft or Google make more money last year?
   Output: [(question='How much profit did Microsoft make last year?', answer=None), (question='How much profit did Google make last year?', answer=None)]
2. Query: What is the capital of France?
   Output: [(question='What is the capital of France?', answer=None)]
3. Query:{search_query}
   Output:
"""

parallel_questions=client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {"role": "system", "content": FanOut_SYSTEM_PROMPT},
        {"role": "user", "content": search_query}
    ],response_format={"type":"json_object"}
)
print("Parallel Questions:", parallel_questions.choices[0].message.content)

parallel_questions=json.loads(parallel_questions.choices[0].message.content)
# print("Parsed Parallel Questions:", parallel_questions)
for question in parallel_questions:
    print("Question:",question['question'])
    

question_docs = []
for question in parallel_questions:
    relevant_docs=retriver.similarity_search(question['question'])
    relevant_docs = [doc.page_content for doc in relevant_docs]
    context= "\n".join(relevant_docs)
    question_docs.append({"question": question['question'], "context": context})

# print("Question Docs:", format("\n".join([f"Question: {pair['question']}\nContext: {pair['context']}" for pair in question_docs])))

SubQuestion_SYSTEM_PROMPT=f""" You are a helpful assistant that can answer complex queries.
    Here is the original question you were asked: {search_query}
    
    Rules:
        -Follow the Output JSON Format.
        -if code in output['answer'] then provide the code.

    Output Json format:
    {{
        "question": "string",
        "answer": "string"
    }}

    And you have split the task into the following questions:
   { "\n".join([f"Question: {question['question']}" for question in parallel_questions])}
    Here are the question and context pairs for each question.
    For each question, generate the question answer pair as a structured output based on the context provided.
    {
    "\n".join([f"Question: {pair['question']}\nContext: {pair['context']}" for pair in question_docs])
    }
    
    Example Output:
    [
       {{ "question": "What is the capital of France?",
        "answer": "The capital of France is Paris."
       }},
        ..// Add more question-answer pairs as needed        
    ]
"""

# print("SubQuestion System Prompt:", SubQuestion_SYSTEM_PROMPT)
final_Question_Answer=client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[
            {"role": "system", "content": SubQuestion_SYSTEM_PROMPT},
            {"role": "user", "content": search_query}
        ],response_format={"type":"json_object"}
    )
print("Chat Response:", final_Question_Answer.choices[0].message.content)

final_Question_Answer=json.loads(final_Question_Answer.choices[0].message.content)
# join all question,answer into a single context
final_context = "\n".join([f"Question: {question['question']}\nAnswer: {question['answer']}" for question in final_Question_Answer])

print("Final Context:", final_context)
final_LLM_SYSTEM_PROMPT =f"""You are a helpful assistant that can answer complex queries.
Here is the original question you were asked: {search_query}

You have split this question up into simpler questions that can be answered in
isolation.
Here are the questions and answers that you've generated
{ "\n".join([f"Question: {pair['question']}\nAnswer: {pair['answer']}" for pair in parallel_questions]) }

Reason about the final answer to the original query based on above questions and
answers
Final Answer:
"""

# print("Final LLM System Prompt:", final_LLM_SYSTEM_PROMPT)
final_query_Answer=client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {"role": "system", "content": final_LLM_SYSTEM_PROMPT},
        {"role": "user", "content": search_query}
    ]
)

print("\nFinal Answer:", final_query_Answer.choices[0].message.content)










# i have used fanout to generate multiple questions based on the original question.<if simple>
# if complex, then i have used {think machine learning}
                                # think macine
                                # think learning
                                # think machine learning
# then i have used the retriever to get relevant chunks for each question.