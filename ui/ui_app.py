import streamlit as st
import requests
import base64
import os

API_URL = "http://localhost:8000"

st.set_page_config(page_title="PolicyAnalyser", page_icon="🚀", layout="wide")

st.title("🚀 PolicyAnalyser - Free Tier Agentic RAG")
st.markdown("Upload legal documents and chat with them using Advanced LangGraph routing, 100% free stack.")

with st.sidebar:
    st.image(os.path.join(os.path.dirname(__file__), "../assets/logo.png"), use_column_width=True)
    st.header("Document Configuration")
    document_id = st.text_input("Document ID", value="DOC_001")
    
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
    if st.button("Index Document") and uploaded_file:
        with st.spinner("Processing and Indexing PDF..."):
            # Base64 encode the file
            base64_pdf = base64.b64encode(uploaded_file.read()).decode("utf-8")
            
            headers = {"document-id": document_id}
            payload = {"file": base64_pdf}
            
            try:
                response = requests.post(f"{API_URL}/upload-documents", json=payload, headers=headers)
                if response.status_code == 200:
                    st.success(f"Successfully Indexed! {response.json().get('chunks_processed')} chunks created.")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")

st.divider()
st.header("Chat with Document")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("references"):
            with st.expander("View Reference Bounding Boxes"):
                st.json(msg["references"])

# Chat Input
if prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Agent is reasoning..."):
            headers = {"document-id": document_id}
            payload = {"question": prompt}
            
            try:
                response = requests.post(f"{API_URL}/ask", json=payload, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer found.")
                    references = data.get("reference_positions", [])
                    
                    st.markdown(answer)
                    if references:
                        with st.expander("View Reference Bounding Boxes"):
                            st.json(references)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "references": references
                    })
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Backend unreachable: {e}")
