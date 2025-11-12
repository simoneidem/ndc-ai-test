# �� RAG MCP Server - Snakk med din lokale kunnskapsbase!

Denne MCP serveren lar deg snakke med Claude Desktop og få svar fra en lokal vektor database med 5 temaer:

- 🍽️ **Norsk matkultur** - lutefisk, pinnekjøtt, fårikål, brunost, rakfisk
- 🎨 **Renessansen** - Leonardo da Vinci, Michelangelo, humanisme, Gutenberg
- ⚛️ **Kvantfysikk** - bølge-partikkel dualitet, Heisenberg, kvantfloking, Schrödinger
- 🌳 **Amazonas** - biodiversitet, avskoging, urfolk, klimapåvirkning
- 🏭 **Industriell revolusjon** - dampmaskinen, jernbane, urbanisering, arbeidsforhold

## 📁 Viktige filer

- **rag_server.py** - MCP server med 10 tools
- **ingestion.py** - Lager vektor database
- **knowledge_base.txt** - Tekstfil med kunnskap
- **vector_db/** - FAISS database

## ✅ Setup (allerede gjort!)

✓ Vektor database opprettet (11 chunks, 1536 dimensjoner)
✓ Claude Desktop config satt opp
✓ Alle pakker installert

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
