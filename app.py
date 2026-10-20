import streamlit as st
from rag import ask_rag

st.set_page_config(
    page_title="Multi-PDF RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("📚 Multi-PDF RAG Chatbot")
st.write("Ask questions about your uploaded PDF documents.")

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User question input
if prompt := st.chat_input("Ask a question about your PDFs..."):
    # Store and render user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query RAG pipeline and display response
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