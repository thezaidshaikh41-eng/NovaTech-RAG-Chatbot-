import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="NovaTech AI Support",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 NovaTech AI Customer Support")
st.caption("Powered by Groq + ChromaDB")

# -----------------------------
# API KEY
# -----------------------------
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Groq API Key not found.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# -----------------------------
# LOAD EMBEDDING MODEL
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# -----------------------------
# LOAD KNOWLEDGE BASE
# -----------------------------
@st.cache_resource
def build_vector_db():

    documents = []

    data_folder = "data"

    if not os.path.exists(data_folder):
        return None

    for file in os.listdir(data_folder):

        if file.endswith(".txt"):

            with open(
                os.path.join(data_folder, file),
                "r",
                encoding="utf-8"
            ) as f:

                documents.append(
                    {
                        "source": file,
                        "text": f.read()
                    }
                )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = []

    for doc in documents:

        pieces = splitter.split_text(doc["text"])

        for piece in pieces:

            chunks.append(
                {
                    "source": doc["source"],
                    "text": piece
                }
            )

    client_db = chromadb.Client()

    collection = client_db.create_collection(
        name="novatech_support"
    )

    texts = [c["text"] for c in chunks]

    embeddings = embedding_model.encode(texts)

    for i, chunk in enumerate(chunks):

        collection.add(
            ids=[str(i)],
            documents=[chunk["text"]],
            embeddings=[embeddings[i].tolist()],
            metadatas=[
                {
                    "source": chunk["source"]
                }
            ]
        )

    return collection


collection = build_vector_db()

if collection is None:
    st.error("No data folder found.")
    st.stop()

# -----------------------------
# CHAT HISTORY
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# USER INPUT
# -----------------------------
question = st.chat_input("Ask anything about NovaTech...")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    query_embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    context = "\n\n".join(results["documents"][0])

    prompt = f"""
You are NovaTech Electronics AI Customer Support Assistant.

Rules:

Answer ONLY using the provided context.

If answer is unavailable reply exactly:

I couldn't find that information in the company knowledge base.

Context:

{context}

Question:

{question}
"""

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "system",
                "content": "You are a professional customer support assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.choices[0].message.content

    with st.chat_message("assistant"):

        st.markdown(answer)

        with st.expander("📄 Sources"):

            for meta in results["metadatas"][0]:
                st.write("•", meta["source"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
