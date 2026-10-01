import os

from dotenv import load_dotenv
from google import genai

from rag.embeddings import create_embeddings


load_dotenv()


# -----------------------------
# Gemini setup
# -----------------------------

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found in .env"
    )


client = genai.Client(
    api_key=api_key
)


MODEL_NAME = "gemini-3.5-flash-lite"


# -----------------------------
# Generate RAG answer
# -----------------------------

def generate_answer(
    question,
    vector_store,
    top_k=3
):

    # 1. Convert question into embedding
    query_embedding = create_embeddings(
        [question]
    )

    # 2. Retrieve relevant chunks
    results = vector_store.search(
        query_embedding,
        top_k=top_k
    )

    # 3. Handle no results
    if not results:

        return {
            "answer": (
                "I could not find relevant information "
                "in the uploaded documents."
            ),
            "sources": []
        }

    # 4. Build context
    context_parts = []

    sources = []

    for result in results:

        chunk = result["chunk"]

        context_parts.append(
            chunk["text"]
        )

        sources.append({
            "page_number": chunk["page_number"],
            "distance": result["distance"]
        })


    context = "\n\n".join(
        context_parts
    )


    # 5. Create RAG prompt

    prompt = f"""
You are LearnMate AI, a learning assistant.

Answer the student's question using ONLY
the information provided in the context below.

Do not use outside knowledge.

If the answer cannot be found in the context,
clearly say:

"I could not find this information in the
uploaded documents."

Explain the answer clearly and in a
student-friendly way.

-------------------------
CONTEXT
-------------------------

{context}

-------------------------
QUESTION
-------------------------

{question}

-------------------------
ANSWER
-------------------------
"""


    # 6. Send prompt to Gemini

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )


    # 7. Return answer + sources

    return {
        "answer": response.text,
        "sources": sources
    }