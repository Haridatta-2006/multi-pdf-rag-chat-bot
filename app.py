import streamlit as st
from ingest import ingest_uploaded_files
from rag import ask_rag

st.set_page_config(
    page_title="Multi-PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "docs_indexed" not in st.session_state:
    st.session_state.docs_indexed = False
if "indexed_file_names" not in st.session_state:
    st.session_state.indexed_file_names = []

# Sidebar: File Upload & Processing
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        index_btn = st.button("Index Documents", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("Clear Chat", use_container_width=True)

    if clear_btn:
        st.session_state.messages = []
        st.rerun()

    if index_btn:
        if uploaded_files:
            with st.spinner("Reading, chunking, and indexing PDFs..."):
                try:
                    total_chunks = ingest_uploaded_files(uploaded_files)
                    st.session_state.docs_indexed = True
                    st.session_state.indexed_file_names = [f.name for f in uploaded_files]
                    st.success(f"Indexed {len(uploaded_files)} PDF(s) ({total_chunks} chunks).")
                except Exception as e:
                    st.error(f"Error during ingestion: {str(e)}")
        else:
            st.warning("Please choose at least one PDF file first.")

    if st.session_state.docs_indexed:
        st.info("Active documents:\n" + "\n".join([f"- {name}" for name in st.session_state.indexed_file_names]))

# Main Chat Interface
st.title("📚 Multi-PDF RAG Chatbot")
st.caption("Ask questions about your uploaded documents.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt input
if prompt := st.chat_input("Ask a question about your PDFs..."):
    # Append user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant answer
    with st.chat_message("assistant"):
        if not st.session_state.docs_indexed:
            warning_msg = "Please upload and click **Index Documents** before asking questions."
            st.warning(warning_msg)
            st.session_state.messages.append({"role": "assistant", "content": warning_msg})
        else:
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
