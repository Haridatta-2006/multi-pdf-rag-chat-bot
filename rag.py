import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from google import genai

load_dotenv()

def ask_rag(query: str):
    pinecone_key = os.getenv("PINECONE_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "multi-pdf-rag")

    if not pinecone_key:
        raise ValueError("PINECONE_API_KEY is not set.")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    # 1. Embed user query
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    query_vector = embeddings.embed_query(query)

    # 2. Search Pinecone
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(index_name)

    search_results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )

    # 3. Extract retrieved context & sources
    context_chunks = []
    sources = set()

    for match in search_results.get("matches", []):
        metadata = match.get("metadata", {})
        text = metadata.get("text", "")
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "")

        if text:
            context_chunks.append(text)
            source_tag = f"{source} (p. {page})" if page else source
            sources.add(source_tag)

    if not context_chunks:
        return "No relevant context found in uploaded PDFs.", []

    context_text = "\n\n---\n\n".join(context_chunks)

    # 4. Construct prompt
    prompt = f"""You are a helpful assistant answering questions based on provided PDF documents.
Use the following context to answer the user's question accurately.
If the answer cannot be determined from the context, state that you don't know.

Context:
{context_text}

Question: {query}

Answer:"""

    # 5. Generate response using Gemini
    client = genai.Client(api_key=gemini_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text, list(sources)
