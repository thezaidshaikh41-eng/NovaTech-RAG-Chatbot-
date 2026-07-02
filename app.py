import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

# Page config
st.set_page_config(page_title="NovaTech RAG Chatbot")
st.title("🤖 NovaTech AI Customer Support")

# Initialize models
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("novatech")
    return embedder, collection

embedder, collection = load_models()

# Groq API Key
groq_api_key = st.secrets.get("GROQ_API_KEY", "")

if not groq_api_key:
    st.error("Please add GROQ_API_KEY in Streamlit Secrets.")
    st.stop()

groq_client = Groq(api_key=groq_api_key)

# User input
query = st.text_input("Ask a question about NovaTech:")

if query:
    try:
        # Create embedding
        query_embedding = embedder.encode(query).tolist()

        # Search vector database
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        context = "\n".join(results["documents"][0])

        prompt = f"""
You are a helpful NovaTech customer support assistant.

Context:
{context}

Question:
{query}

Answer:
"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )

        answer = response.choices[0].message.content

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved Context"):
            st.write(context)

    except Exception as e:
        st.error(f"Error: {e}")
