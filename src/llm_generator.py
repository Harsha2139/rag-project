
import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def get_api_key():
    try:
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except Exception:
        pass

    return os.getenv("GROQ_API_KEY")


def generate_answer(question, context):

    api_key = get_api_key()

    if not api_key:
        raise ValueError(
            "API key not found. Configure GROQ_API_KEY in Streamlit Secrets."
        )

    client = Groq(api_key=api_key)

    prompt = f"""
Answer the question using only the provided document context.

If the answer is not available in the context, say:
"The document does not contain the answer."

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content
