import streamlit as st
import requests

# Docker internal network URL for the FastAPI backend
API_URL = "http://rag-api:8080/rag"

st.set_page_config(page_title="Physics RAG", layout="centered")
st.title("Physics Research Assistant")
st.markdown("Retrieval Augmented Generation (RAG) for High-Energy & Nuclear Physics")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message and message["context"]:
            with st.expander("Retrieved Context"):
                for ctx in message["context"]:
                    st.markdown(f"- {ctx}")

# Process new user input
if prompt := st.chat_input("Enter your physics query..."):
    # Append and display user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Fetch response from FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Querying the RAG pipeline..."):
            try:
                payload = {"query": prompt, "top_k": 5}
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()
                
                data = response.json()
                answer = data.get("answer", "No answer returned.")
                contexts = data.get("context_used", [])

                # Render response and context
                st.markdown(answer)
                with st.expander("Retrieved Context"):
                    for ctx in contexts:
                        st.markdown(f"- {ctx}")
                
                # Append to state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "context": contexts
                })
                
            except requests.exceptions.RequestException as e:
                st.error(f"API Request failed: {e}")