import streamlit as st

from RAG_from_scratch import (
    create_openai_client,
    initialize_chroma_collection,
    load_text_from_file,
    split_text_into_chunks,
    index_text_chunks,
    build_rag_agent,
)
import chromadb


@st.cache_resource
def setup_rag():
    openai_client = create_openai_client()
    chroma_client = chromadb.Client()
    vector_collection = initialize_chroma_collection(chroma_client, "company_docs")

    document_text = load_text_from_file("hr_document.txt")
    chunks = split_text_into_chunks(document_text, "HR Policy")
    index_text_chunks(vector_collection, chunks)

    return build_rag_agent(openai_client, vector_collection)


st.title("Company Document Q&A")
st.caption("Ask questions about HR, product, security, and engineering policies.")

rag_agent = setup_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask a question about company documents..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            answer = rag_agent(query, verbose=False)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
