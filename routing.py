from openai import OpenAI
import os



client=OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

models={"claude","gemini","deepseek","openai"}
user_query=input("Ask the Question:")
SYSTEM_PROMPT=f"""You are a model routing system.
based on the user_query:{user_query}, you have to choose which LLM is best sooted from available models

Available Models:
{"\n".join([model for model in models])}

Example:
Query:what is the solution of 2(32*34)+23/90?
Output: claude
"""
print(SYSTEM_PROMPT)

chat=client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
)
print("Fianl Model:",chat.choices[0].message.content)