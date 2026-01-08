# Raymond Maarloeve LLMServer

> Official language model (LLM) server for the narrator and NPCs in the **Raymond Maarloeve** game project.

![Python](https://img.shields.io/badge/Python-3.12-blue)  
![API](https://img.shields.io/badge/API-Flask-orange)  
![Model](https://img.shields.io/badge/Backend-llama.cpp-lightgrey)  
![Platform](https://img.shields.io/badge/platform-Cross--platform-informational)  
![Docs](https://img.shields.io/badge/docs-Available-blue)

A lightweight REST API for managing local language models used by NPCs and the narrator in the game. Supports multiple model loading, response generation, and dynamic resource management.

## 📚 Documentation
Full project documentation is available at:  
🔗 **[https://raymondmaarloeve.github.io/LLMServer/](https://raymondmaarloeve.github.io/LLMServer/)**  
Main repo:
🔗 **[https://github.com/RaymondMaarloeve/RaymondMaarloeve](https://github.com/RaymondMaarloeve/RaymondMaarloeve)**  

### 🧠 Embeddings & Character Management
- **[Quick Start Guide](QUICKSTART.md)** - Get embeddings running in 5 minutes
- **[Full Embeddings Documentation](EMBEDDINGS.md)** - Complete guide (Polish)
- **[HTTP Examples](LLMServer_embeddings.http)** - API request examples  

## ✨ Features

- 🔁 Supports multiple LLMs simultaneously (`model_id`)
- 🔌 Simple `/chat` endpoint with full conversation history handling
- 🚦 Automatic response termination detection using special tags (`<npc>`, `<human>`, etc.)
- 🧹 Ability to unload models from memory (`/unload`)
- 📂 File browsing via API (`/list-files`)
- 🧠 **Vector database with embeddings** for NPC character contexts
- 🔍 **Semantic search** for automatic character context retrieval
- 🎭 **Character management** endpoints for medieval game NPCs

## 🧩 Technologies

- [Python 3.12](https://www.python.org/)
- [Flask](https://flask.palletsprojects.com/) – REST API
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) – interface for local LLaMA models
- [ChromaDB](https://www.trychroma.com/) – vector database for character embeddings
- [Sentence Transformers](https://www.sbert.net/) – semantic embeddings for NPC contexts
- [PyInstaller](https://pyinstaller.org/) – server binary packaging

## 🚀 Usage

1. Run the server:
   ```bash
   python main.py
   ```

2. Load a model:
   ```json
   POST /load
   {
     "model_id": "npc_village",
     "model_path": "models/ggml-npc-q4.bin",
     "n_ctx": 2048,
     "n_gpu_layers": 16
   }
   ```

3. Send a chat request:
   ```json
   POST /chat
   {
     "model_id": "npc_village",
     "character_id": "kowal_jan",  // automatic context from vector DB
     "messages": [
       {"role": "user", "content": "Hello there!"}
     ]
   }
   ```
   
   Or use semantic search:
   ```json
   POST /chat
   {
     "model_id": "npc_village",
     "use_embedding": true,  // auto-find best matching character
     "messages": [
       {"role": "user", "content": "Who can fix my sword?"}
     ]
   }
   ```

4. Receive the response and display it in-game.

## 🎭 Character Context Management

The server includes a vector database for storing and retrieving NPC character contexts using embeddings.

### Adding Characters

```bash
# Load example characters
python load_characters.py

# List all characters
python load_characters.py list

# Test semantic search
python load_characters.py test
```

### API Example

```json
POST /add-character-context
{
  "character_id": "blacksmith_john",
  "name": "John the Blacksmith",
  "context": "You are John, a grumpy blacksmith who..."
}
```

The character context will be automatically embedded and stored. You can then use it in chat requests:

```json
POST /chat
{
  "model_id": "npc_model",
  "character_id": "blacksmith_john",
  "messages": [
    {"role": "user", "content": "Can you fix my sword?"}
  ]
}
```

## 🛠 Building

To build a standalone version:
```bash
 CMAKE_ARGS="-DGGML_VULKAN=on" uv pip install llama-cpp-python --no-cache
 uv run pyinstaller --onefile --additional-hooks-dir hooks main.py
```

## 🔍 API Endpoints

| Endpoint                    | Description                                        |
|-----------------------------|----------------------------------------------------|
| `/load`                     | Load a model into memory                           |
| `/chat`                     | Generate a response in chat style                  |
| `/unload`                   | Release model resources                            |
| `/status`                   | Check available models and GPU status              |
| `/list-files`               | List files in a specified directory                |
| `/register`                 | Register a model for lazy-loading                  |
| `/add-character-context`    | Add/update character context with embeddings       |
| `/get-character-context`    | Retrieve character context by ID or semantic query |
| `/list-characters`          | List all stored character contexts                 |

---

> The `LLMServer` project is the foundation of narration and NPC behavior in the world of Raymond Maarloeve.  
