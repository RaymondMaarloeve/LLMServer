"""
LLMServer - REST API for managing local language models using llama-cpp-python.

This server provides endpoints for:
- Loading and unloading GGUF language models
- Chat completion with conversation history
- Model registration and lazy-loading
- Server status monitoring
- File browsing

Features:
- Multiple model support with unique IDs
- GPU acceleration support
- Streaming token-by-token responses
- Resource management and model switching
- Comprehensive error handling with tracebacks
"""

import pathlib
import traceback
from flask import Flask, request, jsonify
from llama_cpp import Llama, llama_cpp, load_shared_library
import time

app = Flask(__name__)

# Global dictionary to store loaded language model instances
# Key: model_id (unique identifier), Value: Llama instance
models = {}

# Global dictionary to store registered model paths for lazy-loading
# Key: model_id, Value: path to model file
registered_models = {}


@app.route("/load", methods=["POST"])
def load_model():
    """
    Load a GGUF language model into memory.

    Expected JSON payload:
    {
        "model_id": "unique_identifier",     # Required: ID to reference this model
        "model_path": "/path/to/model.gguf", # Required: Path to the model file
        "n_ctx": 1024,                       # Optional: Context window size (default: 1024)
        "n_parts": -1,                       # Optional: Model parts (-1 auto-detects)
        "seed": 42,                          # Optional: Random seed (default: 42)
        "f16_kv": false,                     # Optional: FP16 key-value caching (default: false)
        "n_gpu_layers": -1                   # Optional: GPU layers (-1 offloads all)
    }

    Returns:
    - 200: Model loaded successfully
    - 400: Validation error or model already loaded
    - 500: Failed to load model

    Response:
    {
        "message": "Model 'model_id' loaded successfully from ...",
        "success": true
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

    # Register the model for future reference if not already registered
    if model_id not in registered_models:
        registered_models[model_id] = model_path

    # Check if model is already loaded
    if model_id in models:
        return jsonify({"message": f"Model with ID '{model_id}' is already loaded.", "success": False}), 400

    # Extract model parameters with defaults
    n_ctx = data.get("n_ctx", 1024)
    n_parts = data.get("n_parts", -1)
    seed = data.get("seed", 42)
    f16_kv = data.get("f16_kv", False)
    n_gpu_layers = data.get("n_gpu_layers", -1)

    # Attempt to load the model using llama-cpp-python
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
    Generate a response using a loaded model with chat message history.

    Expected JSON payload:
    {
        "model_id": "model_identifier",      # Required: Which model to use
        "messages": [                         # Required: Conversation history
            {
                "role": "system",             # Optional: System prompt
                "content": "You are helpful..."
            },
            {
                "role": "user",               # User message
                "content": "Your question..."
            },
            {
                "role": "assistant",          # Previous assistant response
                "content": "My response..."
            }
        ],
        "max_tokens": 500,                    # Optional: Max response tokens (default: 100)
        "temperature": 0.8,                   # Optional: Sampling temperature (default: 0.8)
        "top_p": 0.95,                        # Optional: Nucleus sampling (default: 0.95)
        "n_ctx": 4096,                        # Optional: Context window (default: 1024)
        "n_parts": -1,                        # Optional: Model parts (default: -1)
        "seed": 42,                           # Optional: Random seed (default: 42)
        "f16_kv": false,                      # Optional: FP16 caching (default: false)
        "n_gpu_layers": -1                    # Optional: GPU layers (default: -1)
    }

    Auto-behavior:
    - Unloads other models to free memory
    - Loads the requested model if not in memory
    - Stops generation when encountering special tags: <assistant>, <human>, <npc>, <system>

    Returns:
    - 200: Response generated successfully
    - 400: Validation error
    - 500: Generation failed

    Response:
    {
        "response": "Generated assistant response text",
        "generation_time": 2.345,             # Seconds
        "model_id": "model_identifier",
        "total_tokens": 42,                   # Number of tokens generated
        "success": true
    }
    """
    global models, registered_models
    data = request.get_json()
    if not data:
        return jsonify({"message": "No input data provided.", "success": False}), 400

    model_id = data.get("model_id")
    messages = data.get("messages", [])

    if not model_id or not messages:
        return jsonify({"message": "Missing required parameters: 'model_id' and 'messages'.", "success": False}), 400

    # Check if model is registered
    model_path = registered_models.get(model_id)
    if model_path is None:
        return jsonify({"message": f"Model '{model_id}' is not registered. Register it first using /register.", "success": False}), 400

    # Unload all other models to free memory (single model in memory at a time)
    to_unload = [mid for mid in models if mid != model_id]
    for mid in to_unload:
        try:
            models.pop(mid)
        except Exception:
            pass

    # Load the model if not already in memory
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

    # Validate all messages have required fields and valid roles
    for msg in messages:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            return jsonify({"message": "Invalid message format. Each message must have 'role' and 'content' fields.", "success": False}), 400
        if msg["role"] not in ["system", "user", "assistant"]:
            return jsonify({"message": f"Invalid role: '{msg['role']}'. Must be 'system', 'user', or 'assistant'.", "success": False}), 400

    # Extract generation parameters with defaults
    max_tokens = data.get("max_tokens", 100)
    temperature = data.get("temperature", 0.8)
    top_p = data.get("top_p", 0.95)

    try:
        # Convert message history to prompt format
        formatted_prompt = format_chat_messages(messages)

        # Generate response from the model with streaming
        start_time = time.time()
        generated_text = ""
        total_tokens = 0
        stop_generating = False

        # Process tokens as they stream from the model
        for response in model(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True
        ):
            if "choices" in response and response["choices"]:
                token = response["choices"][0]["text"]
                generated_text += token
                lower_generated_text = generated_text.lower()

                # Check for special tags that signal end of response
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

        # Calculate how long generation took
        generation_time = time.time() - start_time
        return jsonify({
            "response": generated_text.strip(),
            "generation_time": round(generation_time, 3),
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
    Unload a model from memory to free resources.

    Expected JSON payload:
    {
        "model_id": "model_identifier"  # Required: Model to unload
    }

    Returns:
    - 200: Model unloaded successfully
    - 400: Model not found or validation error
    - 500: Failed to unload

    Response:
    {
        "message": "Model 'model_id' has been unloaded successfully.",
        "success": true
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

    # Remove model from memory
    try:
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
    Get server health status and information about loaded models.

    No request body required.

    Returns:
    - 200: Always returns successfully

    Response:
    {
        "healthy": true,                     # Server health status
        "models": ["model_id1", "model_id2"], # Array of loaded model IDs
        "gpu": true                          # GPU acceleration available
    }
    """
    global models

    # Check if GPU acceleration is available
    gpu = False
    try:
        # Try to load the llama.cpp shared library to check GPU support
        p = pathlib.Path(llama_cpp.__file__).parent
        lib = load_shared_library('llama', pathlib.Path(p) / 'lib')
        gpu = bool(lib.llama_supports_gpu_offload())
    except:
        # GPU not available or error loading library
        pass

    return jsonify({
        "healthy": True,
        "models": list(models.keys()),
        "gpu": gpu
    }), 200

@app.route("/list-files", methods=["POST"])
def list_files():
    """
    List all files in a specified directory.

    Expected JSON payload:
    {
        "directory": "/path/to/directory"  # Required: Directory path to list
    }

    Returns:
    - 200: Files listed successfully
    - 400: Directory is not a directory or validation error
    - 404: Directory does not exist
    - 500: Failed to list files

    Response (Success):
    {
        "success": true,
        "files": [
            {
                "name": "model.gguf",
                "path": "/path/to/directory/model.gguf"
            }
        ]
    }

    Response (Error):
    {
        "message": "Directory '/path/to/directory' does not exist.",
        "success": false
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"message": "No input data provided.", "success": False}), 400

    directory = data.get("directory")
    if not directory:
        return jsonify({"message": "Missing required parameter: 'directory'.", "success": False}), 400

    try:
        # Convert directory path to Path object
        dir_path = pathlib.Path(directory)

        # Check if directory exists
        if not dir_path.exists():
            return jsonify({
                "message": f"Directory '{directory}' does not exist.",
                "success": False
            }), 404

        # Check if path is a directory
        if not dir_path.is_dir():
            return jsonify({
                "message": f"'{directory}' is not a directory.",
                "success": False
            }), 400

        # List all files in the directory (not subdirectories)
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
    Register a model ID with its file path for lazy-loading.

    This allows models to be referenced by ID without immediately loading them.
    The model will be loaded when first accessed in a /chat request.

    Expected JSON payload:
    {
        "model_id": "model_identifier",     # Required: Unique ID for the model
        "model_path": "/path/to/model.gguf" # Required: Path to the model file
    }

    Returns:
    - 200: Model registered successfully
    - 400: Validation error (missing parameters)

    Response:
    {
        "message": "Model 'model_id' registered with path '/path/to/model.gguf'.",
        "success": true
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

def format_chat_messages(messages):
    """
    Format a list of chat messages into a single prompt string.

    Converts message history with roles (system, user, assistant) into a formatted
    prompt compatible with various GGUF language models.

    Format:
    <system>
    System prompt content
    </system>

    <human>: User message 1
    <assistant>: Assistant response 1
    <human>: User message 2
    <assistant>:

    Args:
        messages (list): List of message dicts with 'role' and 'content' keys
                        Roles: 'system', 'user', 'assistant'

    Returns:
        str: Formatted prompt string ready for model input

    Example:
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        prompt = format_chat_messages(messages)
        # Returns: "<system>\\nYou are helpful.\\n</system>\\n\\n<human>: Hello\\n<assistant>: Hi there!\\n<human>: How are you?\\n<assistant>: "
    """
    prompt = ""

    # Extract system message if present (used as initial context)
    system_message = None
    for msg in messages:
        if msg["role"] == "system":
            system_message = msg["content"]
            break

    # Add system message at the beginning if present
    if system_message:
        prompt += f"<system>\n{system_message}\n</system>\n\n"

    # Add all messages in conversation order, converting roles to tags
    for msg in messages:
        if msg["role"] == "system":
            # Skip system message as it was already handled at the top
            continue

        if msg["role"] == "user":
            prompt += f"<human>: {msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt += f"<assistant>: {msg['content']}\n"

    # Add the final assistant tag to prompt the model to generate a response
    prompt += "<assistant>: "

    return prompt

if __name__ == '__main__':
    """
    Start the LLMServer Flask application.

    Configuration:
    - host='0.0.0.0': Listen on all network interfaces
    - port=5000: Server port
    - debug=True: Enable debug mode with hot reloading

    For production, change debug=False to disable debug mode and auto-reloading.
    For security, bind to 'localhost' instead of '0.0.0.0' if not exposing over network.

    Access the server at: http://localhost:5000
    """
    app.run(host='0.0.0.0', port=5000, debug=True)