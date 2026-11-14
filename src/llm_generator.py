
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


def get_client():

    api_key = get_api_key()

    if not api_key:
        raise ValueError(
            "API key not found. Configure GROQ_API_KEY in Streamlit Secrets."
        )

    return Groq(api_key=api_key)


def generate_answer(question, context):

    if not context or not context.strip():
        return "The uploaded documents do not contain relevant information to answer this question."

    client = get_client()

    prompt = f"""
You are SALAAR AI, an intelligent multi-document assistant.

Answer the user's question using ONLY the information provided
in the retrieved document context.

IMPORTANT RULES:

1. Search and use information from ALL provided context sections.
2. Information may come from multiple PDF documents.
3. Combine information from different documents when relevant.
4. Do not assume that all information comes from one PDF.
5. Do not invent facts, examples, figures, or explanations.
6. If the context does not support the answer, clearly say:
   "The uploaded documents do not contain sufficient information to answer this question."
7. When possible, mention the PDF filename and page number supporting the answer.
8. Give a clear, structured answer.
9. If the question asks for a comparison, compare only the information
   available in the supplied context.
10. Do not claim to have checked documents or pages that are not included
    in the context.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


def generate_summary(context, document_name="Uploaded Documents"):

    if not context or not context.strip():
        return "No readable document content is available for summarization."

    client = get_client()

    prompt = f"""
You are SALAAR AI, a document summarization assistant.

Create a detailed, well-organized summary using ONLY the supplied
document content.

DOCUMENT NAME:
{document_name}

RULES:

1. Identify the main topics and important subtopics.
2. Preserve important definitions, facts, steps, examples and figures.
3. Organize the summary using headings and bullet points.
4. Do not invent missing information.
5. If multiple documents are provided, keep their information
   distinguishable and combine related topics where appropriate.
6. Mention source filenames and page numbers when available.
7. Do not claim to summarize content that is not provided.

DOCUMENT CONTENT:

{context}

SUMMARY:
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content
