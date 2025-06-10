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

## ✨ Features

- 🔁 Supports multiple LLMs simultaneously (`model_id`)
- 🔌 Simple `/chat` endpoint with full conversation history handling
- 🚦 Automatic response termination detection using special tags (`<npc>`, `<human>`, etc.)
- 🧹 Ability to unload models from memory (`/unload`)
- 📂 File browsing via API (`/list-files`)

## 🧩 Technologies

- [Python 3.12](https://www.python.org/)
- [Flask](https://flask.palletsprojects.com/) – REST API
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) – interface for local LLaMA models
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
     "messages": [
       {"role": "system", "content": "You are a grumpy blacksmith."},
       {"role": "user", "content": "Hello there!"},
       {"role": "assistant", "content": "Hmph. What do you want?"},
       {"role": "user", "content": "Got any gossip?"}
     ]
   }
   ```

4. Receive the response and display it in-game.

## 🛠 Building

To build a standalone version:
```bash
 CMAKE_ARGS="-DGGML_VULKAN=on" uv pip install llama-cpp-python --no-cache
 uv run pyinstaller --onefile --additional-hooks-dir hooks main.py
```

## 🔍 API Endpoints

| Endpoint      | Description                              |
|---------------|------------------------------------------|
| `/load`       | Load a model into memory                 |
| `/chat`       | Generate a response in chat style        |
| `/unload`     | Release model resources                  |
| `/status`     | Check available models and GPU status    |
| `/list-files` | List files in a specified directory      |
| `/register`   | Register a model for lazy-loading        |

---

> The `LLMServer` project is the foundation of narration and NPC behavior in the world of Raymond Maarloeve.  
