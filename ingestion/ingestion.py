"""
RAG Ingestion Pipeline
Leser tekstfil, chunker den, lager embeddings og lagrer i FAISS
"""
import os
from openai import OpenAI
import numpy as np
import faiss
import pickle

# Initialiser OpenAI klient
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chunk_text(text, chunk_size=200, overlap=50):
    """
    Deler tekst opp i chunks med overlapp
    
    Args:
        text: Teksten som skal chunkes
        chunk_size: Antall ord per chunk
        overlap: Antall ord som overlapper mellom chunks
    
    Returns:
        Liste med text chunks
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:  # Ikke legg til tomme chunks
            chunks.append(chunk)
    
    return chunks

def get_embedding(text):
    """
    Henter embedding fra OpenAI
    
    Args:
        text: Teksten som skal embeddes
    
    Returns:
        Embedding vektor (numpy array)
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

def ingest_document(file_path, output_dir="vector_db"):
    """
    Hovedfunksjon for ingestion pipeline
    
    Args:
        file_path: Path til tekstfilen
        output_dir: Mappe hvor vektordatabase lagres
    """
    print(f"📖 Leser fil: {file_path}")
    
    # Les tekstfil
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"📊 Totalt antall ord: {len(text.split())}")
    
    # Chunk teksten
    print("✂️  Chunker tekst...")
    chunks = chunk_text(text)
    print(f"📦 Antall chunks: {len(chunks)}")
    
    # Lag embeddings for hver chunk
    print("🔮 Lager embeddings...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        if i % 5 == 0:
            print(f"  ⏳ Prosesserer chunk {i+1}/{len(chunks)}")
        embedding = get_embedding(chunk)
        embeddings.append(embedding)
    
    # Konverter til numpy array
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    # Opprett FAISS index
    print("🗄️  Oppretter FAISS index...")
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
    index.add(embeddings_array)
    
    # Lag output directory hvis den ikke eksisterer
    os.makedirs(output_dir, exist_ok=True)
    
    # Lagre FAISS index
    index_path = os.path.join(output_dir, "index.faiss")
    faiss.write_index(index, index_path)
    print(f"💾 FAISS index lagret: {index_path}")
    
    # Lagre chunks (vi trenger disse for å returnere faktisk tekst senere)
    chunks_path = os.path.join(output_dir, "chunks.pkl")
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"💾 Chunks lagret: {chunks_path}")
    
    print(f"\n✅ Ingestion fullført!")
    print(f"   - {len(chunks)} chunks prosessert")
    print(f"   - {dimension} dimensjoner per embedding")
    print(f"   - Vektordatabase lagret i: {output_dir}/")
    
    return {
        "num_chunks": len(chunks),
        "embedding_dim": dimension,
        "output_dir": output_dir
    }

if __name__ == "__main__":
    # Sjekk at OpenAI API key er satt
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Feil: OPENAI_API_KEY miljøvariabel er ikke satt!")
        print("Kjør: export OPENAI_API_KEY='din-api-key'")
        exit(1)
    
    # Kjør ingestion
    print("\n" + "="*60)
    print("  RAG INGESTION PIPELINE")
    print("="*60 + "\n")
    
    result = ingest_document("knowledge_base.txt", output_dir="vector_db")
    
    print("\n" + "="*60)
    print("  KLAR TIL Å KJØRE MCP SERVER!")
    print("="*60)
    print("\nKjør nå: python rag_server.py")
    print("Test med: python interactive_client.py\n")
