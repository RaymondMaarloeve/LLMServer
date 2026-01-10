# LLMServer

A lightweight REST API server for managing and running local language models using Flask and llama-cpp-python. Designed for running multiple LLM instances with support for model loading, chat completion, and dynamic resource management.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Usage Examples](#usage-examples)
- [Architecture](#architecture)
- [Development](#development)

## Features

- **Multiple Model Support**: Load and manage multiple language models simultaneously with unique identifiers
- **Chat Completion**: Generate responses using chat-based message format with conversation history
- **Model Registration**: Register models for lazy-loading on-demand
- **Resource Management**: Automatic model unloading to free up memory
- **GPU Acceleration**: Support for GPU offloading with configurable layer distribution
- **File Browser**: List and manage files on the server filesystem via API
- **Streaming Support**: Token-by-token response generation with real-time output
- **Error Handling**: Comprehensive error messages with traceback information
- **Performance Metrics**: Track generation time and token count for each response

## Requirements

- Python 3.12+
- Flask 3.1.0+
- llama-cpp-python
- PyInstaller (for building standalone binaries)

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd LLMServer
```

### 2. Install dependencies

Using `uv` (recommended):

```bash
uv sync
```

Or using pip:

```bash
pip install -r requirements.txt
```

### 3. Build llama-cpp-python with GPU support (optional)

For Vulkan acceleration:

```bash
CMAKE_ARGS="-DGGML_VULKAN=on" uv pip install llama-cpp-python --no-cache
```

For CUDA acceleration:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python --no-cache
```

## Configuration

### Server Settings

Edit the last lines of `main.py` to configure the server:

```python
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

- **host**: Server binding address (default: `0.0.0.0` - accessible from any interface)
- **port**: Server port (default: `5000`)
- **debug**: Debug mode (default: `True` - set to `False` in production)

### Model Parameters

When loading models, you can configure:

- **n_ctx**: Context window size (default: 1024)
- **n_parts**: Number of model parts (default: -1, auto-detect)
- **seed**: Random seed for generation (default: 42)
- **f16_kv**: Use FP16 key-value caching (default: False)
- **n_gpu_layers**: Number of layers to offload to GPU (default: -1, offload all)

## API Endpoints

### 1. `/load` (POST)

Load a language model into memory.

**Request:**

```json
{
  "model_id": "my_model",
  "model_path": "/path/to/model.gguf",
  "n_ctx": 4096,
  "n_parts": -1,
  "seed": 42,
  "f16_kv": true,
  "n_gpu_layers": -1
}
```

**Response (Success):**

```json
{
  "message": "Model 'my_model' loaded successfully from /path/to/model.gguf.",
  "success": true
}
```

**Response (Error):**

```json
{
  "message": "Failed to load model 'my_model': <error details>\ntrace: <traceback>",
  "success": false
}
```

**Status Code:** 200 (success), 400 (validation error), 500 (server error)

---

### 2. `/chat` (POST)

Generate a response using a loaded model with chat message history.

**Request:**

```json
{
  "model_id": "my_model",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is 2+2?"
    },
    {
      "role": "assistant",
      "content": "2+2 equals 4."
    },
    {
      "role": "user",
      "content": "Tell me more."
    }
  ],
  "max_tokens": 500,
  "temperature": 0.8,
  "top_p": 0.95,
  "n_ctx": 4096,
  "n_parts": -1,
  "seed": 42,
  "f16_kv": true,
  "n_gpu_layers": -1
}
```

**Message Roles:**
- `system`: System prompt (optional, defines model behavior)
- `user`: User message
- `assistant`: Assistant response from previous turns

**Response (Success):**

```json
{
  "response": "The sum of 2 and 2 is 4, which is a fundamental arithmetic fact.",
  "generation_time": 2.345,
  "model_id": "my_model",
  "total_tokens": 23,
  "success": true
}
```

**Response (Error):**

```json
{
  "message": "Chat completion failed for model 'my_model': <error details>\ntrace: <traceback>",
  "success": false
}
```

**Status Code:** 200 (success), 400 (validation error), 500 (server error)

**Notes:**
- The model is automatically loaded if not already in memory
- Only one model can be in memory at a time (other models are unloaded)
- Generation stops automatically when encountering tags: `<assistant>`, `<human>`, `<npc>`, `<system>`, `</assistant>`, `</human>`, `</npc>`, `</system>`

---

### 3. `/unload` (POST)

Unload a model from memory to free up resources.

**Request:**

```json
{
  "model_id": "my_model"
}
```

**Response (Success):**

```json
{
  "message": "Model 'my_model' has been unloaded successfully.",
  "success": true
}
```

**Response (Error):**

```json
{
  "message": "Failed to unload model 'my_model': <error details>",
  "success": false
}
```

**Status Code:** 200 (success), 400 (validation error), 500 (server error)

---

### 4. `/register` (POST)

Register a model ID with a file path for later lazy-loading.

**Request:**

```json
{
  "model_id": "my_model",
  "model_path": "/path/to/model.gguf"
}
```

**Response (Success):**

```json
{
  "message": "Model 'my_model' registered with path '/path/to/model.gguf'.",
  "success": true
}
```

**Response (Error):**

```json
{
  "message": "Missing required parameters: 'model_id' and 'model_path'.",
  "success": false
}
```

**Status Code:** 200 (success), 400 (validation error)

---

### 5. `/status` (GET)

Get the current server status and loaded models information.

**Request:**

No body required.

**Response:**

```json
{
  "healthy": true,
  "models": ["my_model", "another_model"],
  "gpu": true
}
```

**Response Fields:**
- `healthy`: Server health status
- `models`: Array of currently loaded model IDs
- `gpu`: Whether GPU acceleration is available

**Status Code:** 200

---

### 6. `/list-files` (POST)

List all files in a specified directory.

**Request:**

```json
{
  "directory": "/path/to/directory"
}
```

**Response (Success):**

```json
{
  "success": true,
  "files": [
    {
      "name": "model1.gguf",
      "path": "/path/to/directory/model1.gguf"
    },
    {
      "name": "model2.gguf",
      "path": "/path/to/directory/model2.gguf"
    }
  ]
}
```

**Response (Error - Directory not found):**

```json
{
  "message": "Directory '/path/to/directory' does not exist.",
  "success": false
}
```

**Status Code:** 200 (success), 404 (not found), 400 (validation error), 500 (server error)

## Usage Examples

### Example 1: Load a Model

```bash
curl -X POST http://localhost:5000/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "chat_model",
    "model_path": "/models/model.gguf",
    "n_ctx": 4096,
    "n_gpu_layers": -1
  }'
```

### Example 2: Generate a Response

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "chat_model",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello, how are you?"
      }
    ],
    "max_tokens": 100,
    "temperature": 0.8,
    "top_p": 0.95
  }'
```

### Example 3: Check Server Status

```bash
curl -X GET http://localhost:5000/status
```

### Example 4: List Available Models

```bash
curl -X POST http://localhost:5000/list-files \
  -H "Content-Type: application/json" \
  -d '{
    "directory": "/models"
  }'
```

### Example 5: Unload a Model

```bash
curl -X POST http://localhost:5000/unload \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "chat_model"
  }'
```

## Architecture

### Core Components

#### Global State

```python
models = {}              # Loaded model instances (model_id -> Llama)
registered_models = {}  # Registered model paths (model_id -> path)
```

#### Key Functions

1. **load_model()**: Load a GGUF model into memory with specified parameters
2. **chat()**: Generate responses using loaded models with conversation history
3. **unload_model()**: Remove a model from memory
4. **status()**: Return server health and model information
5. **register_model()**: Register a model path without loading
6. **list_files()**: Browse server filesystem
7. **format_chat_messages()**: Convert message history to prompt format

#### Message Formatting

Messages are formatted into a prompt compatible with various LLM models:

```
<system>
You are a helpful assistant.
</system>

<human>: Hello, how are you?
<assistant>: I'm doing well, thank you!
<human>: Tell me about yourself.
<assistant>: 
```

### Model Management

- **Single Model in Memory**: Only one model can be loaded at a time to optimize memory usage
- **Auto-Loading**: Models are automatically loaded when referenced in chat requests
- **Auto-Unloading**: Other loaded models are automatically unloaded when switching models
- **Registration**: Models can be pre-registered without loading to reduce startup time

### Error Handling

All endpoints return structured error responses with:
- Human-readable error message
- Success flag for easy parsing
- Full Python traceback for debugging

## Development

### Running the Server

```bash
python main.py
```

The server will start on `http://0.0.0.0:5000`

### Building a Standalone Executable

```bash
CMAKE_ARGS="-DGGML_VULKAN=on" uv pip install llama-cpp-python --no-cache
uv run pyinstaller --onefile --additional-hooks-dir hooks main.py
```

The executable will be in `dist/main.exe` (Windows) or `dist/main` (Linux/macOS)

### Debug Mode

The server runs with `debug=True` by default, which enables:
- Hot reloading on code changes
- Detailed error messages
- Interactive debugger on errors

For production, set `debug=False` in the last line of `main.py`.

### Performance Considerations

- **GPU Acceleration**: Use `n_gpu_layers=-1` to offload all layers to GPU for best performance
- **Context Window**: Larger `n_ctx` values increase memory usage exponentially
- **Token Streaming**: Responses are streamed token-by-token for better user experience
- **Model Switching**: Unloading and loading models incurs overhead; minimize switching

## Troubleshooting

### Issue: Module not found errors

**Solution**: Ensure all dependencies are installed:
```bash
uv sync
```

### Issue: CUDA/GPU not detected

**Solution**: Rebuild llama-cpp-python with GPU support:
```bash
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python --no-cache
```

### Issue: Out of memory errors

**Solution**:
- Reduce `n_ctx` size
- Reduce `n_gpu_layers` to offload fewer layers
- Use a smaller model file

### Issue: Slow response generation

**Solution**:
- Check if GPU acceleration is available (`/status` endpoint)
- Increase `n_gpu_layers` to use more GPU
- Reduce `n_ctx` if not needed

## License

This project is part of the Raymond Maarloeve game project.

## Support

For issues, questions, or contributions, please refer to the main repository at:
https://github.com/RaymondMaarloeve/RaymondMaarloeve
