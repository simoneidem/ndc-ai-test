# 🚀 RAG MCP Server - Snakk med din lokale kunnskapsbase!

Denne MCP serveren lar deg snakke med Claude Desktop og få svar fra en lokal vektor database med 5 temaer:

- 🍽️ **Norsk matkultur** - lutefisk, pinnekjøtt, fårikål, brunost, rakfisk
- 🎨 **Renessansen** - Leonardo da Vinci, Michelangelo, humanisme, Gutenberg
- ⚛️ **Kvantfysikk** - bølge-partikkel dualitet, Heisenberg, kvantfloking, Schrödinger
- 🌳 **Amazonas** - biodiversitet, avskoging, urfolk, klimapåvirkning
- 🏭 **Industriell revolusjon** - dampmaskinen, jernbane, urbanisering, arbeidsforhold

## 📁 Prosjektstruktur

```
├── ingestion/          # Embedding og vektor database
│   ├── ingestion.py
│   ├── knowledge_base.txt
│   └── vector_db/
├── mcp-server/         # MCP server med 10 tools
│   └── rag_server.py
├── mcp-client/         # MCP klient (Claude Desktop)
└── requirements.txt    # Python avhengigheter
```

## 🔧 Setup

### 1. Installer avhengigheter
```bash
pip install -r requirements.txt
```

### 2. Generer vektor database (hvis ikke gjort)
```bash
cd ingestion
python ingestion.py
```

### 3. Konfigurer Claude Desktop

Rediger `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rag-knowledge-base": {
      "command": "/path/to/.venv/bin/python",
      "args": ["/path/to/mcp-server/rag_server.py"],
      "env": {
        "OPENAI_API_KEY": "your-api-key"
      }
    }
  }
}
```

## 🚀 START HER - Bruk med Claude Desktop

### 1. Restart Claude Desktop
Command+Q og åpne på nytt

### 2. Spør Claude!

**Eksempler:**
- "Hva er fårikål?"
- "Fortell meg om Leonardo da Vinci"
- "Forklar kvantfloking"
- "Hvorfor er Amazonas viktig?"
- "Hva var den industrielle revolusjonen?"

Claude velger automatisk riktig tool og henter svar fra lokal database!

### 3. Sjekk tools
Spør: "Hvilke tools har du tilgang til?"

## 🔧 De 10 MCP Tools

**Tema-spesifikke:**
1. search_norwegian_food
2. search_renaissance
3. search_quantum_physics
4. search_amazon_rainforest
5. search_industrial_revolution

**Generelle:**
6. search_documents
7. get_topics_overview
8. get_database_info
9. list_all_chunks
10. get_chunk_by_index

## 🐛 Feilsøking

**Claude ser ikke tools?**
- Restart Claude helt
- Sjekk `~/Library/Application Support/Claude/claude_desktop_config.json`

**Feil ved tool bruk?**
- Sjekk Console.app for feilmeldinger

---

**Det er det! Restart Claude og test! 🎉**
