import pathlib
import traceback
from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import chromadb
from chromadb.utils import embedding_functions
import json

app = Flask(__name__)

# Initialize ChromaDB for character context embeddings
chroma_client = chromadb.Client()
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
try:
    character_contexts = chroma_client.get_or_create_collection(
        name="character_contexts",
        embedding_function=embedding_function
    )
except Exception as e:
    print(f"Warning: Could not initialize ChromaDB collection: {e}")
    character_contexts = None

# Global dictionary to store model instances
# Each key is a unique string model_id and the value is a dict with 'model' and 'tokenizer'
models = {}

# Global dictionary to store registered model_id -> model_path mappings
registered_models = {}

# Global dictionary to store memory capsules (embedding tensors) for each model
# key: model_id, value: torch.Tensor with pre-computed embeddings
memory_capsules = {}


def load_memory_capsule(model_path: str, model_id: str, device: str = "cuda"):
    """
    Wczytuje kapsułę pamięci z pre-computed embeddingami (.pt file).
    
    Args:
        model_path: Ścieżka do folderu modelu
        model_id: ID modelu
        device: Urządzenie do załadowania tensorów (cuda/cpu)
    """
    global memory_capsules
    
    try:
        model_dir = pathlib.Path(model_path)
        if model_dir.is_file():
            model_dir = model_dir.parent
        
        capsule_path = model_dir / "memory_capsule.pt"
        meta_path = model_dir / "memory_capsule_meta.json"
        
        if capsule_path.exists():
            print(f"🧠 Wczytywanie kapsuły EMBEDDINGÓW dla modelu '{model_id}' z: {capsule_path}")
            
            # Wczytaj tensor z embeddingami
            embeddings = torch.load(capsule_path, map_location=device)
            memory_capsules[model_id] = embeddings
            
            # Wczytaj metadane jeśli istnieją
            if meta_path.exists():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                print(f"✅ Kapsuła wczytana: {meta.get('num_tokens', '?')} tokenów, wymiar {meta.get('embedding_dim', '?')}")
            else:
                print(f"✅ Kapsuła wczytana: shape {embeddings.shape}")
            
            return True
        else:
            print(f"ℹ️ Brak kapsuły embeddingów dla modelu '{model_id}' w: {model_dir}")
            return False
            
    except Exception as e:
        print(f"⚠️ Błąd podczas wczytywania kapsuły pamięci: {e}")
        return False


@app.route("/load", methods=["POST"])
def load_model():
    """
    Load a model using PyTorch/Transformers with support for embedding capsules.

    Expected JSON payload:
    {
         "model_id": "unique_model_identifier",   // required
         "model_path": "path/to/model",           // required: HuggingFace model path or local folder
         "device": "cuda",                        // optional: cuda/cpu (default: auto-detect)
         "torch_dtype": "float16",                // optional: float16/float32/auto (default: auto)
         "use_4bit": false,                       // optional: use 4-bit quantization (default: false)
         "max_length": 2048                       // optional: max context length (default: 2048)
    }
    """
    global models, registered_models
    data = request.get_json()
    if not data:
        return jsonify({"message": "No input data provided.", "success": False}), 400

    model_id = data.get("model_id")
    model_path = data.get("model_path")

    if not model_id or not model_path:
        return jsonify({"message": "Missing required parameters: 'model_id' and 'model_path'.", "success": False}), 400

    # Register model_id and model_path
    if model_id not in registered_models:
        registered_models[model_id] = model_path

    if model_id in models:
        return jsonify({"message": f"Model with ID '{model_id}' is already loaded.", "success": False}), 400

    # Parameters
    device = data.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype_str = data.get("torch_dtype", "auto")
    use_4bit = data.get("use_4bit", False)
    max_length = data.get("max_length", 2048)
    
    # LoRA adapter support
    is_lora = data.get("is_lora", False)
    base_model_path = data.get("base_model_path", None)
    merge_adapter = data.get("merge_adapter", True)  # Domyślnie merge dla wydajności

    try:
        print(f"🔧 Ładowanie modelu '{model_id}' z: {model_path}")
        
        # Dla LoRA: załaduj tokenizer z adaptera (ma referencję do base)
        if is_lora and base_model_path:
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Określ dtype
        if torch_dtype_str == "float16":
            torch_dtype = torch.float16
        elif torch_dtype_str == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = "auto"
        
        # Załaduj model
        if is_lora and base_model_path:
            print(f"   📦 Wykryto LoRA adapter. Base model: {base_model_path}")
            from peft import PeftModel
            
            # Załaduj base model
            if use_4bit:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    dtype=torch_dtype
                )
                if device != "auto":
                    base_model = base_model.to(device)
            
            # Załaduj adapter
            print(f"   🔗 Ładowanie LoRA adaptera z: {model_path}")
            model = PeftModel.from_pretrained(base_model, model_path)
            
            # Merge adapter dla wydajności (opcjonalnie)
            if merge_adapter:
                print(f"   🔀 Mergowanie adaptera z base modelem...")
                model = model.merge_and_unload()
                print(f"   ✅ Adapter zmergowany")
            
        elif use_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch_dtype
            )
            if device != "auto":
                model = model.to(device)
        
        models[model_id] = {
            "model": model,
            "tokenizer": tokenizer,
            "device": str(model.device),
            "max_length": max_length
        }
        
        # Wczytaj kapsułę embeddingów jeśli istnieje
        device_for_capsule = str(model.device).split(':')[0]  # cuda:0 -> cuda
        load_memory_capsule(model_path, model_id, device=device_for_capsule)
        
        print(f"✅ Model '{model_id}' załadowany na {model.device}")
        
        return jsonify({
            "message": f"Model '{model_id}' loaded successfully.",
            "success": True,
            "has_memory_capsule": model_id in memory_capsules,
            "device": str(model.device)
        }), 200
        
    except Exception as e:
        return jsonify({
            "message": f"Failed to load model '{model_id}': {str(e)}\ntrace: {traceback.format_exc()}",
            "success": False
        }), 500

@app.route("/chat", methods=["POST"])
def chat():
    """
    Generate responses using PyTorch model with DIRECT EMBEDDING INJECTION support.

    Expected JSON payload:
    {
        "model_id": "unique_model_identifier",
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ],
        "max_tokens": 100,
        "temperature": 0.8,
        "top_p": 0.95,
        "use_memory_capsule": true  // Wstrzykuje pre-computed embeddingi!
    }
    """
    global models, registered_models, character_contexts, memory_capsules
    data = request.get_json()
    if not data:
        return jsonify({"message": "No input data provided.", "success": False}), 400

    model_id = data.get("model_id")
    messages = data.get("messages", [])
    use_memory_capsule = data.get("use_memory_capsule", True)

    if not model_id or not messages:
        return jsonify({"message": "Missing required parameters: 'model_id' and 'messages'.", "success": False}), 400

    # Sprawdź czy model jest załadowany
    if model_id not in models:
        return jsonify({"message": f"Model '{model_id}' is not loaded. Load it first using /load.", "success": False}), 400

    model_dict = models[model_id]
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    device = model.device

    # Sformatuj wiadomości do promptu
    formatted_prompt = format_chat_messages(messages)
    
    # Tokenizuj prompt użytkownika
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)
    
    # 🚀 MAGIA: Wstrzyknięcie kapsuły jako embeddingi!
    if use_memory_capsule and model_id in memory_capsules:
        print(f"🧠 Wstrzykiwanie kapsuły embeddingów do kontekstu...")
        
        # Pobierz pre-computed embeddingi z kapsuły
        capsule_embeddings = memory_capsules[model_id].to(device)
        
        # Pobierz embeddingi dla pytania użytkownika
        with torch.no_grad():
            query_embeddings = model.get_input_embeddings()(input_ids)
        
        # ⚡ CONCATENATION: Sklej kapsułę + pytanie
        combined_embeddings = torch.cat([capsule_embeddings, query_embeddings], dim=1)
        
        print(f"   Kapsuła: {capsule_embeddings.shape}")
        print(f"   Zapytanie: {query_embeddings.shape}")
        print(f"   Połączone: {combined_embeddings.shape}")
        
        # Użyj inputs_embeds zamiast input_ids!
        inputs_embeds = combined_embeddings
        input_ids_for_generate = None
    else:
        # Bez kapsuły - normalny tryb
        inputs_embeds = None
        input_ids_for_generate = input_ids

    # Parametry generowania
    max_tokens = data.get("max_tokens", 100)
    temperature = data.get("temperature", 0.8)
    top_p = data.get("top_p", 0.95)

    try:
        start_time = time.time()
        
        # 🎯 GENEROWANIE z inputs_embeds!
        with torch.no_grad():
            if inputs_embeds is not None:
                # Tryb z kapsułą - używamy inputs_embeds
                outputs = model.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            else:
                # Tryb normalny - używamy input_ids
                outputs = model.generate(
                    input_ids_for_generate,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
        
        # Dekoduj odpowiedź
        if inputs_embeds is not None:
            # Z inputs_embeds - dekoduj tylko wygenerowaną część
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # Normalny tryb - usuń prompt z odpowiedzi
            generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        generation_time = time.time() - start_time
        
        return jsonify({
            "response": generated_text.strip(),
            "generation_time": round(generation_time, 3),
            "model_id": model_id,
            "used_embedding_capsule": use_memory_capsule and model_id in memory_capsules,
            "success": True
        }), 200
        
    except Exception as e:
        return jsonify({
            "message": f"Chat completion failed: {str(e)}\ntrace:{traceback.format_exc()}",
            "success": False
        }), 500

@app.route("/unload", methods=["POST"])
def unload_model():
    """
    Unload (delete) the specified LLaMA model to free up resources.

    Expected JSON payload:
    {
         "model_id": "unique_model_identifier"   // required: specifies which model to unload
    }
    """
    global models
    data = request.get_json()
    if not data:
        return jsonify({"message": "No input data provided.", "success": False}), 400

    model_id = data.get("model_id")
    if not model_id:
        return jsonify({"message": "Missing required parameter: 'model_id'.", "success": False}), 400

    if model_id not in models:
        return jsonify({"message": f"Model with ID '{model_id}' is not loaded.", "success": False}), 400

    try:
        # Unload the model by removing it from the dictionary. The garbage collector
        # will later reclaim the memory.
        models.pop(model_id)
        return jsonify({
            "message": f"Model '{model_id}' has been unloaded successfully.",
            "success": True
        }), 200
    except Exception as e:
        return jsonify({
            "message": f"Failed to unload model '{model_id}': {str(e)}",
            "success": False
        }), 500

@app.route("/status", methods=["GET"])
def status():
    """
    Returns status of loaded models.
    """
    global models

    model_info = {}
    for model_id, model_dict in models.items():
        model_info[model_id] = {
            "device": model_dict.get("device", "unknown"),
            "has_capsule": model_id in memory_capsules
        }
    
    return jsonify({
        "healthy": True,
        "models": list(models.keys()),
        "model_details": model_info,
        "gpu_available": torch.cuda.is_available()
    }), 200

@app.route("/list-files", methods=["POST"])
def list_files():
    """
    List files in the specified directory.

    Expected JSON payload:
    {
        "directory": "path/to/directory"  // required: path to the directory to list
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"message": "No input data provided.", "success": False}), 400

    directory = data.get("directory")
    if not directory:
        return jsonify({"message": "Missing required parameter: 'directory'.", "success": False}), 400

    try:
        dir_path = pathlib.Path(directory)

        if not dir_path.exists():
            return jsonify({
                "message": f"Directory '{directory}' does not exist.",
                "success": False
            }), 404

        if not dir_path.is_dir():
            return jsonify({
                "message": f"'{directory}' is not a directory.",
                "success": False
            }), 400

        # Get only files
        files = [
            {
                "name": item.name,
                "path": str(item)
            }
            for item in dir_path.iterdir()
            if item.is_file()
        ]

        return jsonify({
            "success": True,
            "files": files
        }), 200

    except Exception as e:
        return jsonify({
            "message": f"Failed to list files: {str(e)}",
            "success": False
        }), 500

@app.route("/register", methods=["POST"])
def register_model():
    """
    Register a model_id with a model_path for later use.

    Expected JSON payload:
    {
        "model_id": "unique_model_identifier",   // required
        "model_path": "path/to/ggml-model.bin"   // required
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"message": "No input data provided.", "success": False}), 400

    model_id = data.get("model_id")
    model_path = data.get("model_path")

    if not model_id or not model_path:
        return jsonify({"message": "Missing required parameters: 'model_id' and 'model_path'.", "success": False}), 400

    registered_models[model_id] = model_path
    return jsonify({
        "message": f"Model '{model_id}' registered with path '{model_path}'.",
        "success": True
    }), 200

@app.route("/add-character-context", methods=["POST"])
def add_character_context():
    """
    Add or update a character's context (system prompt) to the vector database.

    Expected JSON payload:
    {
        "character_id": "unique_character_identifier",  // required: e.g., "blacksmith_john"
        "name": "John the Blacksmith",                  // required: character's display name
        "context": "You are John, a blacksmith..."      // required: full character context/system prompt
    }
    """
    global character_contexts
    
    if character_contexts is None:
        return jsonify({
            "message": "Character context system is not initialized.",
            "success": False
        }), 500
    
    data = request.get_json()
    if not data:
        return jsonify({"message": "No input data provided.", "success": False}), 400

    character_id = data.get("character_id")
    context = data.get("context")

    if not character_id or not context:
        return jsonify({
            "message": "Missing required parameters: 'character_id' and 'context'.",
            "success": False
        }), 400

    try:
        # Prepare metadata - only store name and character_id
        name = data.get("name")
        if not name:
            return jsonify({
                "message": "Missing required parameter: 'name'.",
                "success": False
            }), 400
        
        metadata = {
            "character_id": character_id,
            "name": name
        }

        # Add to ChromaDB (will update if character_id already exists)
        character_contexts.upsert(
            documents=[context],
            metadatas=[metadata],
            ids=[character_id]
        )

        return jsonify({
            "message": f"Character context for '{character_id}' added/updated successfully.",
            "success": True
        }), 200
    except Exception as e:
        return jsonify({
            "message": f"Failed to add character context: {str(e)}",
            "success": False
        }), 500

@app.route("/get-character-context", methods=["POST"])
def get_character_context():
    """
    Retrieve a character's context by character_id or search for similar contexts.

    Expected JSON payload:
    {
        "character_id": "blacksmith_john",  // optional: exact character lookup
        "query": "blacksmith who makes weapons",  // optional: semantic search query
        "n_results": 1  // optional: number of similar results to return (default: 1)
    }
    """
    global character_contexts
    
    if character_contexts is None:
        return jsonify({
            "message": "Character context system is not initialized.",
            "success": False
        }), 500
    
    data = request.get_json()
    if not data:
        return jsonify({"message": "No input data provided.", "success": False}), 400

    character_id = data.get("character_id")
    query = data.get("query")
    n_results = data.get("n_results", 1)

    try:
        if character_id:
            # Direct lookup by ID
            result = character_contexts.get(ids=[character_id])
            if result["documents"]:
                return jsonify({
                    "success": True,
                    "character_id": character_id,
                    "name": result["metadatas"][0].get("name") if result["metadatas"] else None,
                    "context": result["documents"][0]
                }), 200
            else:
                return jsonify({
                    "message": f"Character '{character_id}' not found.",
                    "success": False
                }), 404
        
        elif query:
            # Semantic search
            results = character_contexts.query(
                query_texts=[query],
                n_results=n_results
            )
            
            if results["documents"] and results["documents"][0]:
                characters = []
                for i, doc in enumerate(results["documents"][0]):
                    characters.append({
                        "character_id": results["ids"][0][i],
                        "name": results["metadatas"][0][i].get("name") if results["metadatas"] else None,
                        "context": doc,
                        "distance": results["distances"][0][i] if "distances" in results else None
                    })
                
                return jsonify({
                    "success": True,
                    "results": characters
                }), 200
            else:
                return jsonify({
                    "message": "No matching characters found.",
                    "success": False
                }), 404
        else:
            return jsonify({
                "message": "Either 'character_id' or 'query' must be provided.",
                "success": False
            }), 400

    except Exception as e:
        return jsonify({
            "message": f"Failed to retrieve character context: {str(e)}",
            "success": False
        }), 500

@app.route("/list-characters", methods=["GET"])
def list_characters():
    """
    List all stored character contexts.
    """
    global character_contexts
    
    if character_contexts is None:
        return jsonify({
            "message": "Character context system is not initialized.",
            "success": False
        }), 500
    
    try:
        all_data = character_contexts.get()
        
        characters = []
        if all_data["ids"]:
            for i, char_id in enumerate(all_data["ids"]):
                characters.append({
                    "character_id": char_id,
                    "name": all_data["metadatas"][i].get("name") if all_data["metadatas"] else None,
                    "context": all_data["documents"][i] if all_data["documents"] else ""
                })
        
        return jsonify({
            "success": True,
            "count": len(characters),
            "characters": characters
        }), 200
    except Exception as e:
        return jsonify({
            "message": f"Failed to list characters: {str(e)}",
            "success": False
        }), 500

def format_chat_messages(messages):
    """
    Format a list of chat messages into a single prompt string.
    Uses a format compatible with various LLaMA models.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys

    Returns:
        A formatted prompt string
    """
    prompt = ""

    # Extract system message if present
    system_message = None
    for msg in messages:
        if msg["role"] == "system":
            system_message = msg["content"]
            break

    # Start with system message if available
    if system_message:
        prompt += f"<system>\n{system_message}\n</system>\n\n"

    # Add conversation history
    for msg in messages:
        if msg["role"] == "system":
            continue  # Skip system message as it was already handled

        if msg["role"] == "user":
            prompt += f"<human>: {msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt += f"<assistant>: {msg['content']}\n"

    # Add final assistant prompt
    prompt += "<assistant>: "

    return prompt

if __name__ == '__main__':
    # On Windows, use_reloader=False prevents OSError with select.select()
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)