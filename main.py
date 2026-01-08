import pathlib
import traceback
from flask import Flask, request, jsonify
from llama_cpp import Llama
import time  # Add this import at the top with other imports
import chromadb
from chromadb.utils import embedding_functions

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

# Global dictionary to store multiple LLaMA model instances.
# Each key is a unique string model_id and the value is its Llama instance.
models = {}

# Global dictionary to store registered model_id -> model_path mappings
registered_models = {}


@app.route("/", methods=["GET"])
def index():
    """Health check endpoint."""
    return jsonify({
        "message": "LLMServer is running",
        "loaded_models": list(models.keys()),
        "endpoints": ["/load", "/chat", "/complete", "/embeddings", "/search_context", "/load_characters"]
    }), 200


@app.route("/load", methods=["POST"])
def load_model():
    """
    Load a LLaMA model using llama-cpp-python under a specific model ID.

    Expected JSON payload:
    {
         "model_id": "unique_model_identifier",   // required: used to reference the model later
         "model_path": "path/to/ggml-model.bin",  // required: path to the model file
         "n_ctx": 1024,                           // optional: context window size (default: 1024)
         "n_parts": -1,                           // optional: number of model parts, -1 auto-detects parts
         "seed": 42,                              // optional: RNG seed (default: 42)
         "f16_kv": false,                         // optional: whether to use fp16 key-value caching
         "n_gpu_layers": -1                       // optional: number of layers to offload to GPU, -1 for all (default: -1)
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

    # Register model_id and model_path if not already registered
    if model_id not in registered_models:
        registered_models[model_id] = model_path

    if model_id in models:
        return jsonify({"message": f"Model with ID '{model_id}' is already loaded.", "success": False}), 400

    # Use provided parameters or default values.
    n_ctx = data.get("n_ctx", 1024)
    n_parts = data.get("n_parts", -1)
    seed = data.get("seed", 42)
    f16_kv = data.get("f16_kv", False)
    n_gpu_layers = data.get("n_gpu_layers", -1)  # -1 offloads all layers to GPU

    try:
        model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_parts=n_parts,
            seed=seed,
            f16_kv=f16_kv,
            n_gpu_layers=n_gpu_layers,
        )
        models[model_id] = model
        return jsonify({
            "message": f"Model '{model_id}' loaded successfully from {model_path}.",
            "success": True
        }), 200
    except Exception as e:
        return jsonify({
            "message": f"Failed to load model '{model_id}': {str(e)}\ntrace: {traceback.format_exc()}",
            "success": False
        }), 500

@app.route("/chat", methods=["POST"])
def chat():
    """
    Generate responses using a chat-based format with user and assistant messages.

    Expected JSON payload:
    {
        "model_id": "unique_model_identifier",  // required: specifies which model to use
        "character_id": "blacksmith_john",       // optional: character context from vector DB
        "messages": [                            // required: array of message objects
            {"role": "system", "content": "You are a helpful assistant."}, // optional system message
            {"role": "user", "content": "Hello, how are you?"},            // user messages
            {"role": "assistant", "content": "I'm doing well, thank you!"}, // assistant messages
            {"role": "user", "content": "Tell me about yourself."}         // typically ends with user
        ],
        "max_tokens": 100,                       // optional: maximum tokens to generate (default: 100)
        "temperature": 0.8,                      // optional: sampling temperature (default: 0.8)
        "top_p": 0.95,                           // optional: nucleus sampling top_p (default: 0.95)
        "use_embedding": true                    // optional: auto-retrieve context from embeddings (default: false)
    }

    Returns just the generated assistant response text.
    """
    global models, registered_models, character_contexts
    data = request.get_json()
    if not data:
        return jsonify({"message": "No input data provided.", "success": False}), 400

    model_id = data.get("model_id")
    messages = data.get("messages", [])
    character_id = data.get("character_id")
    use_embedding = data.get("use_embedding", False)

    if not model_id or not messages:
        return jsonify({"message": "Missing required parameters: 'model_id' and 'messages'.", "success": False}), 400

    # Auto-retrieve character context from embeddings if requested
    if character_contexts is not None and (character_id or use_embedding):
        try:
            context_to_add = None
            
            if character_id:
                # Direct lookup by character_id
                result = character_contexts.get(ids=[character_id])
                if result["documents"] and result["documents"]:
                    context_to_add = result["documents"][0]
            elif use_embedding and messages:
                # Semantic search based on the last user message
                user_messages = [msg for msg in messages if msg.get("role") == "user"]
                if user_messages:
                    last_user_msg = user_messages[-1]["content"]
                    results = character_contexts.query(
                        query_texts=[last_user_msg],
                        n_results=1
                    )
                    if results["documents"] and results["documents"][0]:
                        context_to_add = results["documents"][0][0]
            
            # Inject context as system message if found and no system message exists
            if context_to_add:
                has_system = any(msg.get("role") == "system" for msg in messages)
                if not has_system:
                    messages.insert(0, {"role": "system", "content": context_to_add})
                    
        except Exception as e:
            # Don't fail the request if embedding retrieval fails
            print(f"Warning: Failed to retrieve character context: {e}")

    # Check if the model is registered
    model_path = registered_models.get(model_id)
    if model_path is None:
        return jsonify({"message": f"Model '{model_id}' is not registered. Register it first using /register.", "success": False}), 400

    # Unload all models except the requested one
    to_unload = [mid for mid in models if mid != model_id]
    for mid in to_unload:
        try:
            models.pop(mid)
        except Exception:
            pass  # Ignore unload errors

    # Load the requested model if not loaded
    model = models.get(model_id)
    if model is None:
        n_ctx = data.get("n_ctx", 1024)
        n_parts = data.get("n_parts", -1)
        seed = data.get("seed", 42)
        f16_kv = data.get("f16_kv", False)
        n_gpu_layers = data.get("n_gpu_layers", -1)
        try:
            model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_parts=n_parts,
                seed=seed,
                f16_kv=f16_kv,
                n_gpu_layers=n_gpu_layers,
            )
            models[model_id] = model
        except Exception as e:
            return jsonify({
                "message": f"Failed to load model '{model_id}': {str(e)}\ntrace: {traceback.format_exc()}",
                "success": False
            }), 500

    # 3. Validate messages
    for msg in messages:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            return jsonify({"message": "Invalid message format. Each message must have 'role' and 'content' fields.", "success": False}), 400
        if msg["role"] not in ["system", "user", "assistant"]:
            return jsonify({"message": f"Invalid role: '{msg['role']}'. Must be 'system', 'user', or 'assistant'.", "success": False}), 400

    # Optional parameters with default values
    max_tokens = data.get("max_tokens", 100)
    temperature = data.get("temperature", 0.8)
    top_p = data.get("top_p", 0.95)

    try:
        # Format the messages into a prompt
        formatted_prompt = format_chat_messages(messages)
        
        # Generate response using the loaded LLaMA model
        start_time = time.time()
        generated_text = ""
        total_tokens = 0
        stop_generating = False
        for response in model(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True # token-by-token response
        ):
            if "choices" in response and response["choices"]:
                token = response["choices"][0]["text"]
                generated_text += token
                lower_generated_text = generated_text.lower()
                tags = ["<assistant>", "<human>", "<npc>", "<system>", "</assistant>", "</human>", "</npc>", "</system>"]
                for tag in tags:
                    if tag in lower_generated_text:
                        tag_pos = lower_generated_text.find(tag)
                        generated_text = generated_text[:tag_pos]
                        stop_generating = True
                        break
                if stop_generating:
                    break
                total_tokens += 1

        # Calculate generation time
        generation_time = time.time() - start_time
        return jsonify({
            "response": generated_text.strip(),
            "generation_time": round(generation_time, 3),  # Round to 3 decimal places
            "model_id": model_id,
            "total_tokens": total_tokens,
            "success": True
        }), 200
    except Exception as e:
        return jsonify({
            "message": f"Chat completion failed for model '{model_id}': {str(e)}\ntrace:{traceback.format_exc()}",
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
    Returns ids of all loaded models.
    """
    global models

    gpu = False
    try:
        p = pathlib.Path(llama_cpp.__file__).parent
        lib = load_shared_library('llama', pathlib.Path(p) / 'lib')
        gpu = bool(lib.llama_supports_gpu_offload())
    except:
        pass
    
    return jsonify({
        "healthy": True,
        "models": list(models.keys()),
        "gpu": gpu
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
    app.run(host='0.0.0.0', port=5000, debug=False)