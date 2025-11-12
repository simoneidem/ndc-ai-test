from fastmcp import FastMCP
import numpy as np
from pathlib import Path
import pickle
from typing import List, Dict
import os
from openai import OpenAI
import faiss

mcp = FastMCP(name="RAG MCP Server",
              instructions="This server provides tools for querying a local RAG vector database with FAISS.")

# Konfigurasjon - tilpass disse til din RAG database
VECTOR_DB_DIR = "/Users/simoneidem/Projects/Evidi/NDC AI 2025/MCP Eksemepl dag 1/vector_db"
FAISS_INDEX_PATH = os.path.join(VECTOR_DB_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(VECTOR_DB_DIR, "chunks.pkl")

# Initialiser OpenAI klient for embeddings
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class FAISSRag:
    """RAG system basert på FAISS"""
    
    def __init__(self, index_path: str, chunks_path: str):
        self.index_path = index_path
        self.chunks_path = chunks_path
        self.index = None
        self.chunks = None
        self.load_data()
    
    def load_data(self):
        """Laster FAISS index og chunks fra disk"""
        if Path(self.index_path).exists():
            self.index = faiss.read_index(self.index_path)
            print(f"✅ Loaded FAISS index with {self.index.ntotal} vectors")
        else:
            print(f"⚠️  Warning: {self.index_path} not found")
            # Opprett tom index som fallback
            self.index = faiss.IndexFlatL2(1536)  # text-embedding-3-small dimension
        
        if Path(self.chunks_path).exists():
            with open(self.chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            print(f"✅ Loaded {len(self.chunks)} chunks")
        else:
            print(f"⚠️  Warning: {self.chunks_path} not found")
            self.chunks = []
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Henter embedding fra OpenAI"""
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Søker i FAISS index
        
        Args:
            query_embedding: Query embedding vector
            top_k: Antall resultater å returnere
        
        Returns:
            Liste med relevante chunks
        """
        if self.index.ntotal == 0 or len(self.chunks) == 0:
            return []
        
        # FAISS søk (returnerer distances og indices)
        query_vector = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        
        # Bygg resultat liste
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):  # Sikre at index er gyldig
                results.append({
                    "index": int(idx),
                    "distance": float(distance),
                    "similarity": float(1 / (1 + distance)),  # Konverter L2 distance til similarity score
                    "text": self.chunks[idx]
                })
        
        return results
    
    def search_with_text(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Søker med tekst ved å embedde den først"""
        query_embedding = self.get_embedding(query_text)
        return self.search(query_embedding, top_k)

# Initialiser RAG system
rag_system = FAISSRag(
    index_path=FAISS_INDEX_PATH,
    chunks_path=CHUNKS_PATH
)

@mcp.tool
def search_documents(query: str, top_k: int = 5) -> dict:
    """
    Søker i kunnskapsdatabasen som inneholder informasjon om:
    - Norsk matkultur (lutefisk, pinnekjøtt, fårikål, brunost, rakfisk)
    - Renessansen i Europa (Leonardo da Vinci, Michelangelo, humanisme)
    - Kvantfysikk (bølge-partikkel dualitet, Heisenberg, Schrödinger's katt)
    - Brasiliansk regnskog (Amazonas, biodiversitet, avskoging)
    - Den industrielle revolusjonen (dampmaskinen, jernbane, urbanisering)
    
    Args:
        query: Søketekst - f.eks. "Hva er fårikål?", "Fortell om Leonardo da Vinci", "Hva er kvantfloking?"
        top_k: Antall resultater å returnere (standard 5)
    
    Returns:
        Dictionary med søkeresultater
    """
    try:
        results = rag_system.search_with_text(query, top_k=top_k)
        
        return {
            "query": query,
            "num_results": len(results),
            "results": results
        }
    except Exception as e:
        return {"error": f"Feil ved søk: {str(e)}"}

@mcp.tool
def search_with_embedding(query_embedding: List[float], top_k: int = 5) -> dict:
    """
    Søker i den lokale dokument databasen med en ferdig embeddet query.
    
    Args:
        query_embedding: Query embedding som liste med float verdier
        top_k: Antall resultater å returnere (standard 5)
    
    Returns:
        Dictionary med søkeresultater
    """
    try:
        # Konverter til numpy array
        query_vec = np.array(query_embedding)
        
        # Søk i databasen
        results = rag_system.search(query_vec, top_k=top_k)
        
        return {
            "query_embedding_dim": len(query_embedding),
            "num_results": len(results),
            "results": results
        }
    except Exception as e:
        return {"error": f"Feil ved søk: {str(e)}"}

@mcp.tool
def get_chunk_by_index(index: int) -> dict:
    """
    Henter en spesifikk chunk basert på index.
    
    Args:
        index: Index for chunken du vil hente
    
    Returns:
        Dictionary med chunk data
    """
    try:
        if index < 0 or index >= len(rag_system.chunks):
            return {"error": f"Index {index} er utenfor range. Totalt {len(rag_system.chunks)} chunks."}
        
        result = {
            "index": index,
            "text": rag_system.chunks[index],
        }
        
        if rag_system.metadata and index < len(rag_system.metadata):
            result["metadata"] = rag_system.metadata[index]
        
        return result
    except Exception as e:
        return {"error": f"Feil ved henting av chunk: {str(e)}"}

@mcp.tool
def get_database_info() -> dict:
    """
    Henter informasjon om RAG databasen.
    
    Returns:
        Dictionary med database statistikk
    """
    return {
        "num_vectors": rag_system.index.ntotal if rag_system.index else 0,
        "num_chunks": len(rag_system.chunks) if rag_system.chunks else 0,
        "embedding_dimension": rag_system.index.d if rag_system.index else 0,
        "index_type": "FAISS IndexFlatL2",
        "vector_db_dir": VECTOR_DB_DIR,
        "embedding_model": "text-embedding-3-small"
    }

@mcp.tool
def list_all_chunks(limit: int = 100) -> dict:
    """
    Lister alle chunks i databasen (begrenset).
    
    Args:
        limit: Maksimalt antall chunks å returnere (standard 100)
    
    Returns:
        Dictionary med chunks
    """
    total_chunks = len(rag_system.chunks)
    limited_chunks = rag_system.chunks[:limit]
    
    chunks_with_index = [
        {"index": i, "text": chunk[:200] + "..." if len(chunk) > 200 else chunk}
        for i, chunk in enumerate(limited_chunks)
    ]
    
    return {
        "total_chunks": total_chunks,
        "returned_chunks": len(chunks_with_index),
        "chunks": chunks_with_index
    }

@mcp.tool
def search_norwegian_food(dish_name: str = None, search_query: str = None) -> dict:
    """
    Søker spesifikt etter informasjon om norsk mat og matkultur.
    Databasen inneholder info om: lutefisk, pinnekjøtt, fårikål, brunost, rakfisk.
    
    Args:
        dish_name: Navn på rett (f.eks. "lutefisk", "fårikål", "brunost")
        search_query: Fritekst søk om norsk mat (f.eks. "julemat", "fermentert fisk")
    
    Returns:
        Dictionary med informasjon om norske retter
    """
    try:
        query = dish_name if dish_name else search_query
        if not query:
            return {"error": "Du må spesifisere enten dish_name eller search_query"}
        
        results = rag_system.search_with_text(f"norsk mat matkultur {query}", top_k=3)
        return {
            "query": query,
            "category": "Norsk matkultur",
            "num_results": len(results),
            "results": results
        }
    except Exception as e:
        return {"error": f"Feil ved søk: {str(e)}"}

@mcp.tool
def search_renaissance(topic: str = None, person: str = None) -> dict:
    """
    Søker etter informasjon om renessansen i Europa.
    Databasen inneholder info om: Leonardo da Vinci, Michelangelo, humanisme, Gutenberg.
    
    Args:
        topic: Tema (f.eks. "humanisme", "boktrykkerkunst", "kunst")
        person: Person (f.eks. "Leonardo da Vinci", "Michelangelo")
    
    Returns:
        Dictionary med informasjon om renessansen
    """
    try:
        query = person if person else topic
        if not query:
            return {"error": "Du må spesifisere enten topic eller person"}
        
        results = rag_system.search_with_text(f"renessansen europa {query}", top_k=3)
        return {
            "query": query,
            "category": "Renessansen",
            "num_results": len(results),
            "results": results
        }
    except Exception as e:
        return {"error": f"Feil ved søk: {str(e)}"}

@mcp.tool
def search_quantum_physics(concept: str) -> dict:
    """
    Søker etter informasjon om kvantfysikk og moderne fysikk.
    Databasen inneholder info om: bølge-partikkel dualitet, Heisenberg, kvantfloking, 
    Schrödinger's katt, kvantedatamaskiner.
    
    Args:
        concept: Konsept (f.eks. "usikkerhetsprinsipp", "kvantfloking", "superposisjon")
    
    Returns:
        Dictionary med informasjon om kvantfysikk
    """
    try:
        results = rag_system.search_with_text(f"kvantfysikk kvantemekanikk {concept}", top_k=3)
        return {
            "query": concept,
            "category": "Kvantfysikk",
            "num_results": len(results),
            "results": results
        }
    except Exception as e:
        return {"error": f"Feil ved søk: {str(e)}"}

@mcp.tool
def search_amazon_rainforest(topic: str) -> dict:
    """
    Søker etter informasjon om Amazonas-regnskogen og brasiliansk biodiversitet.
    Databasen inneholder info om: biodiversitet, jaguaren, avskoging, klimapåvirkning, urfolk.
    
    Args:
        topic: Tema (f.eks. "biodiversitet", "avskoging", "jaguar", "urfolk")
    
    Returns:
        Dictionary med informasjon om Amazonas
    """
    try:
        results = rag_system.search_with_text(f"amazonas regnskog brasil {topic}", top_k=3)
        return {
            "query": topic,
            "category": "Amazonas-regnskogen",
            "num_results": len(results),
            "results": results
        }
    except Exception as e:
        return {"error": f"Feil ved søk: {str(e)}"}

@mcp.tool
def search_industrial_revolution(topic: str) -> dict:
    """
    Søker etter informasjon om den industrielle revolusjonen.
    Databasen inneholder info om: dampmaskinen, tekstilindustri, jernbane, 
    urbanisering, arbeidsforhold, miljøkonsekvenser.
    
    Args:
        topic: Tema (f.eks. "dampmaskinen", "jernbane", "arbeidsforhold", "urbanisering")
    
    Returns:
        Dictionary med informasjon om den industrielle revolusjonen
    """
    try:
        results = rag_system.search_with_text(f"industrielle revolusjonen {topic}", top_k=3)
        return {
            "query": topic,
            "category": "Den industrielle revolusjonen",
            "num_results": len(results),
            "results": results
        }
    except Exception as e:
        return {"error": f"Feil ved søk: {str(e)}"}

@mcp.tool
def get_topics_overview() -> dict:
    """
    Returnerer en oversikt over alle tilgjengelige temaer i databasen.
    
    Returns:
        Dictionary med alle temaer og nøkkelord
    """
    return {
        "available_topics": {
            "1. Norsk matkultur": {
                "description": "Tradisjonelle norske retter og matkultur",
                "keywords": ["lutefisk", "pinnekjøtt", "fårikål", "brunost", "rakfisk"],
                "tool": "search_norwegian_food"
            },
            "2. Renessansen i Europa": {
                "description": "Kulturell bevegelse på 1300-1600-tallet",
                "keywords": ["Leonardo da Vinci", "Michelangelo", "humanisme", "Gutenberg"],
                "tool": "search_renaissance"
            },
            "3. Kvantfysikk": {
                "description": "Moderne fysikk og kvantemekanikk",
                "keywords": ["bølge-partikkel", "Heisenberg", "kvantfloking", "Schrödinger"],
                "tool": "search_quantum_physics"
            },
            "4. Amazonas-regnskogen": {
                "description": "Brasiliansk regnskog og biodiversitet",
                "keywords": ["biodiversitet", "avskoging", "jaguar", "urfolk", "klimapåvirkning"],
                "tool": "search_amazon_rainforest"
            },
            "5. Den industrielle revolusjonen": {
                "description": "Industrialisering på 1700-1800-tallet",
                "keywords": ["dampmaskinen", "jernbane", "urbanisering", "arbeidsforhold"],
                "tool": "search_industrial_revolution"
            }
        },
        "general_search": "search_documents",
        "total_topics": 5
    }

@mcp.resource("rag://stats")
def get_stats() -> str:
    """Ressurs som returnerer statistikk om RAG databasen"""
    info = get_database_info()
    return f"""RAG Database Statistics (FAISS):
- Number of vectors: {info['num_vectors']}
- Number of chunks: {info['num_chunks']}
- Embedding dimension: {info['embedding_dimension']}
- Index type: {info['index_type']}
- Embedding model: {info['embedding_model']}
- Database location: {info['vector_db_dir']}

Available Topics:
1. Norsk matkultur (lutefisk, pinnekjøtt, fårikål, brunost, rakfisk)
2. Renessansen i Europa (Leonardo da Vinci, Michelangelo, humanisme)
3. Kvantfysikk (bølge-partikkel dualitet, Heisenberg, kvantfloking)
4. Brasiliansk regnskog (Amazonas, biodiversitet, avskoging)
5. Den industrielle revolusjonen (dampmaskinen, jernbane, urbanisering)
"""

if __name__ == "__main__":
    # Kjør serveren
    mcp.run()
