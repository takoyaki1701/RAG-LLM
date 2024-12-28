import textwrap
from RAG import RAG
import google.generativeai as genai
from dotenv import load_dotenv
import os
import torch
load_dotenv()

api_key = os.getenv("Gemini_API")
file_path = 'data/DemoData.pdf'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"You are running on {device}")

def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

def LLM(user_input):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    message = f"""
"You are an AI assistant that answers questions.",
"You are a very honest person, never make up answer from what you do not know. When and only when you do not know the answer, say exactly I DO NOT KNOW!".
"content": {user_input}
"""
    response = model.generate_content(message).text
    return response


if __name__ == "__main__":
    query = input("Input something to find:")
    llm_response = LLM(query)

    if bool("I DO NOT KNOW!" in llm_response):
        assistant_message, rag_output = RAG(file_path, query)
        query += assistant_message
        rag_response = LLM(query)
        print("\n___________________________RAG answer_____________________________\n")
        print_wrapped(rag_response)
        print("\n__________________________________________________________________\n")

    else:
        print_wrapped(llm_response) 
