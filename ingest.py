import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

# Load environment variables from .env
load_dotenv()

def ingest_pdfs():
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created '{data_dir}' folder. Please place your PDFs inside it.")
        return

    print("Loading PDFs from directory...")
    loader = PyPDFDirectoryLoader(data_dir)
    documents = loader.load()
    
    if not documents:
        print("No PDFs found in the 'data/' folder. Please add at least one PDF.")
        return

    print(f"Loaded {len(documents)} page(s) from the PDFs.")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} text chunks.")

    # Initialize Hugging Face embeddings (all-MiniLM-L6-v2)
    print("Loading Hugging Face embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Connect to Pinecone using the official client
    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=os.getenv("pcsk_96rdh_MsaskQmqAUkaweftdxPdbuDK64THsko8QwDM1w3pQbKQc76ZoMQkWR97F5TxoVM"))
    index = pc.Index("multi-pdf-rag")

    print("Generating embeddings and uploading to Pinecone...")
    
    # Process and upsert vectors in batches
    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_query(chunk.page_content)
        index.upsert(
            vectors=[
                {
                    "id": f"chunk-{i}",
                    "values": vector,
                    "metadata": {"text": chunk.page_content, "source": chunk.metadata.get("source", "unknown")}
                }
            ]
        )
    
    print("Ingestion complete! Chunks successfully stored in Pinecone.")

if __name__ == "__main__":
    ingest_pdfs()
