
import os
from dotenv import load_dotenv
from groq import Groq

# Load API key from .env
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("API key not found. Check your .env file.")

client = Groq(api_key=api_key)


def generate_answer(question, context):

    prompt = f"""
    Answer the question using only the provided context.

    If the answer is not available in the context,
    say that the document does not contain the answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


if __name__ == "__main__":

    question = "What is a while loop?"

    context = """
    A while loop repeatedly executes statements
    as long as its condition evaluates to True.
    It stops when the condition becomes False.
    """

    answer = generate_answer(question, context)

    print("\nAI Generated Answer:")
    print(answer)
