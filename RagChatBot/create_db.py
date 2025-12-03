# ingest.py
import os
from typing import List

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()
client = OpenAI()

CHROMA_PATH = "chroma_db"
DATA_PATH = "data"
COLLECTION_NAME = "docs"


def load_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif ext == ".pdf":
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        print(f"Skipping unsupported file type: {path}")
        return ""


def embed_texts(texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]


def main():
    if not os.path.isdir(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} folder not found. Create it and add some .txt or .pdf files."
        )

   
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


    collection = chroma_client.create_collection(COLLECTION_NAME)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )

    file_list = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH)]
    file_list = [f for f in file_list if os.path.isfile(f)]

    doc_id_counter = 0

    for file_path in file_list:
        print(f"Processing: {file_path}")
        text = load_text_from_file(file_path)
        if not text.strip():
            print("  -> No text found, skipping.")
            continue

        chunks = text_splitter.split_text(text)
        print(f"  -> {len(chunks)} chunks")

       
        batch_size = 32
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            embeddings = embed_texts(batch_chunks)

            ids = [f"doc_{doc_id_counter + j}" for j in range(len(batch_chunks))]
            metadatas = [
                {
                    "source": os.path.basename(file_path),
                    "chunk_index": i + j,
                }
                for j in range(len(batch_chunks))
            ]

            collection.add(
                ids=ids,
                documents=batch_chunks,
                metadatas=metadatas,
                embeddings=embeddings,
            )

            doc_id_counter += len(batch_chunks)

    print("Ingestion complete!")
    print(f"Total chunks stored: {doc_id_counter}")


if __name__ == "__main__":
    main()
