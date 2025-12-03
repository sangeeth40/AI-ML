# app.py
import os
from typing import List

import chromadb
from chromadb.config import Settings
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "docs"


def get_chroma_collection():
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    return chroma_client.get_or_create_collection(COLLECTION_NAME)


def embed_query(text: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return response.data[0].embedding


def retrieve_context(query: str, k: int = 4):
    collection = get_chroma_collection()
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return list(zip(documents, metadatas, distances))


def build_prompt(question: str, context_chunks: List[str]) -> str:
    context_text = "\n\n".join(context_chunks)
    prompt = f"""
You are a helpful RAG chatbot. Use ONLY the information in the context below to answer the user's question.
If the answer is not in the context, say you don't know and suggest how the user could improve their documents.

Context:
{context_text}

Question: {question}

Answer:
"""
    return prompt.strip()


def generate_answer(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for question answering over documents."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def main():
    st.set_page_config(page_title="Simple RAG Chatbot", page_icon="🤖")
    st.title("RAG Chatbot")
    st.write("Ask questions based on the documents in your `data/` folder.")

    if not os.path.isdir(CHROMA_PATH):
        st.warning("Vector database not found. Run `python create_db.py` first to index your documents.")
        st.stop()

    question = st.text_input("Enter your question:")
    top_k = st.slider("Number of chunks to retrieve", min_value=2, max_value=8, value=4, step=1)

    if st.button("Ask") and question.strip():
        with st.spinner("Retrieving relevant context and generating answer..."):
            results = retrieve_context(question, k=top_k)

            if not results:
                st.error("No results found in the database. Try re-ingesting your data.")
                return

            context_chunks = [doc for (doc, meta, dist) in results]
            prompt = build_prompt(question, context_chunks)
            answer = generate_answer(prompt)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved Chunks (Context)")
        for idx, (doc, meta, dist) in enumerate(results):
            with st.expander(f"Chunk {idx+1} | Source: {meta.get('source')} | Distance: {dist:.4f}"):
                st.write(doc)


if __name__ == "__main__":
    main()
