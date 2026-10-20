import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from google import genai

# Load environment variables
load_dotenv()

def ask_rag(query: str):
    # 1. Initialize Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Embed the user's question
    query_vector = embeddings.embed_query(query)
    
    # 3. Connect to Pinecone and search for similar chunks
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("multi-pdf-rag")
    
    search_results = index.query(
        vector=query_vector,
        top_k=3,  # Retrieve top 3 most relevant chunks
        include_metadata=True
    )
    
    # 4. Extract text chunks and sources
    context_chunks = []
    sources = set()
    
    for match in search_results.get("matches", []):
        metadata = match.get("metadata", {})
        text = metadata.get("text", "")
        source = metadata.get("source", "unknown")
        
        context_chunks.append(text)
        sources.add(os.path.basename(source))
        
    context_text = "\n\n---\n\n".join(context_chunks)
    
    # 5. Construct prompt for Gemini
    prompt = f"""
    You are a helpful assistant answering questions based on provided PDF documents.
    Use the following pieces of retrieved context to answer the user's question. 
    If you don't know the answer based on the context, state that you don't know. Do not make things up.

    Context:
    {context_text}

    Question: {query}

    Answer:
    """
    
    # 6. Generate response using gemini-3.6-flash
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    
    return response.text, list(sources)

if __name__ == "__main__":
    # Test query
    test_question = "What is the main topic of the documents?"
    print(f"Asking: {test_question}\n")
    answer, sources = ask_rag(test_question)
    print("Answer:")
    print(answer)
    print(f"\nSources: {sources}")