import os
from io import BytesIO
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

load_dotenv()

def get_pinecone_index():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "multi-pdf-rag")
    if not api_key:
        raise ValueError("PINECONE_API_KEY is not set in environment variables.")
    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)

def ingest_uploaded_files(uploaded_files) -> int:
    """Reads PDF files from Streamlit uploader, chunks them, and stores embeddings in Pinecone."""
    if not uploaded_files:
        return 0

    all_texts = []
    
    # 1. Extract text from each uploaded PDF
    for uploaded_file in uploaded_files:
        pdf_reader = PdfReader(BytesIO(uploaded_file.read()))
        file_name = uploaded_file.name
        
        for page_idx, page in enumerate(pdf_reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                all_texts.append({
                    "text": text,
                    "metadata": {"source": file_name, "page": page_idx + 1}
                })

    if not all_texts:
        return 0

    # 2. Split extracted text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    chunks = []
    for doc in all_texts:
        sub_chunks = text_splitter.split_text(doc["text"])
        for chunk in sub_chunks:
            chunks.append({
                "text": chunk,
                "metadata": doc["metadata"]
            })

    # 3. Create HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    index = get_pinecone_index()

    # 4. Upsert chunks in batches
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectors = []
        for j, item in enumerate(batch):
            vector_id = f"chunk-{i + j}"
            vector_vals = embeddings.embed_query(item["text"])
            vectors.append({
                "id": vector_id,
                "values": vector_vals,
                "metadata": {
                    "text": item["text"],
                    "source": item["metadata"]["source"],
                    "page": item["metadata"]["page"]
                }
            })
        index.upsert(vectors=vectors)

    return len(chunks)
