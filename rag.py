from google import genai
from google.genai import types
import chromadb

client = genai.Client(api_key="")

db = chromadb.PersistentClient(path="./rag-db")

collection = db.get_or_create_collection(name = "documents")


def chunk_text(text, chunk_size, overlap):
    """Split the text into chunks with size 500 and overlap 100"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start+= chunk_size - overlap
    return chunks

def embed_text(text):
    response = client.models.embed_content(
        model = "gemini-embedding-001",
        contents = text
    )
    return response.embeddings[0].values

def similarity_search(text, collection, k=3):
    vector = embed_text(text)
    result = collection.query(
        query_embeddings = [vector],
        n_results=k,
    )
    return result["documents"][0]



text = open("ai_report.txt").read()
chunks = chunk_text(text, chunk_size=500, overlap=100)

existing = collection.get()
if len(existing["ids"]) == 0:
    for i, chunk in enumerate(chunks):
       vector = embed_text(chunk)
       collection.add(
           documents= [chunk],
           embeddings= [vector],
           ids = [f"chunk_{i}"],
       )
       print(f"Stored chunk {i + 1}/{len(chunks)}", flush=True)
else:
    print(" Chunk Exists, so skipping", flush=True)

question = "Major countries where healthcare advancements made"
answer_chunks = similarity_search(question, collection)

context = "\n\n---\n\n".join(answer_chunks)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents = question,
    config= types.GenerateContentConfig(
        system_instruction= f"""You are a helpful assistant, use the context and provide answers.
                            Do not use outside knowledge.
                            --- CONTEXT START ---
                            {context}
                            --- CONTEXT END ---"""
    )
)

print(response.text)