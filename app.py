import streamlit as st
from ingestion import ingest_uploaded_files
from rag import ask_rag

st.set_page_config(
    page_title="Multi-PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

# Sidebar: File Upload & Processing
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if st.button("Index Documents", type="primary"):
        if uploaded_files:
            with st.spinner("Processing and indexing PDFs..."):
                try:
                    total_chunks = ingest_uploaded_files(uploaded_files)
                    st.success(f"Successfully processed {len(uploaded_files)} PDF(s) into {total_chunks} chunks!")
                except Exception as e:
                    st.error(f"Error during ingestion: {str(e)}")
        else:
            st.warning("Please select at least one PDF file first.")

st.title("📚 Multi-PDF RAG Chatbot")
st.caption("Ask questions about your uploaded documents.")

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your PDFs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents & generating answer..."):
            try:
                answer, sources = ask_rag(prompt)
                response_content = answer
                if sources:
                    sources_str = ", ".join(sources)
                    response_content += f"\n\n---\n**Sources cited:** `{sources_str}`"

                st.markdown(response_content)
                st.session_state.messages.append({"role": "assistant", "content": response_content})
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
