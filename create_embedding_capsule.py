"""
🧠 AKT 1: "PAKOWALNIA" - Generator Kapsuły Pamięci z prawdziwymi embeddingami

Ten skrypt tworzy plik .pt z pre-computed embeddingami z instructions.txt.
Nie potrzebujesz pełnego modelu - tylko tokenizer i warstwa embeddingu.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pathlib
import json

def create_embedding_capsule(
    model_path: str,
    instructions_file: str,
    output_folder: str,
    use_4bit: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    is_lora: bool = False,
    base_model_path: str = None
):
    """
    Tworzy kapsułę pamięci z prawdziwymi embeddingami.
    
    Args:
        model_path: Ścieżka do modelu HuggingFace lub lokalnego folderu (lub adaptera LoRA)
        instructions_file: Plik .txt z instrukcjami (np. geralt.txt)
        output_folder: Folder gdzie zapisać kapsułę (np. qwen-3b-distilled_latest)
        use_4bit: Czy użyć kwantyzacji 4-bit (oszczędność RAM)
        device: cuda/cpu
        is_lora: Czy model_path to adapter LoRA
        base_model_path: Ścieżka do base modelu (wymagane jeśli is_lora=True)
    """
    
    print("=" * 70)
    print("🧠 PAKOWALNIA - Generator Kapsuły Pamięci z Embeddingami")
    print("=" * 70)
    
    # Wczytaj tekst instrukcji
    print(f"\n📖 Wczytywanie instrukcji z: {instructions_file}")
    with open(instructions_file, 'r', encoding='utf-8') as f:
        instructions_text = f.read()
    
    print(f"   📝 Długość: {len(instructions_text)} znaków, {len(instructions_text.split())} słów")
    
    # Załaduj tokenizer
    if is_lora and base_model_path:
        print(f"\n🔧 Ładowanie tokenizera z base modelu: {base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    else:
        print(f"\n🔧 Ładowanie tokenizera z: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Tokenizuj tekst
    print(f"🔢 Tokenizacja...")
    tokens = tokenizer.encode(instructions_text, return_tensors="pt")
    num_tokens = tokens.shape[1]
    print(f"   ✅ Liczba tokenów: {num_tokens}")
    
    # Załaduj TYLKO warstwę embeddingu (nie cały model!)
    print(f"\n🧠 Ładowanie warstwy embeddingu...")
    
    if is_lora and base_model_path:
        print(f"   📦 Wykryto LoRA adapter")
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
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device
            )
        
        # Załaduj adapter i merge
        print(f"   🔗 Ładowanie adaptera z: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
        print(f"   🔀 Mergowanie adaptera...")
        model = model.merge_and_unload()
        
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
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device
        )
    
    # Wyciągnij warstwę embeddingu
    embedding_layer = model.get_input_embeddings()
    
    print(f"   ✅ Wymiary embeddingu: {embedding_layer.embedding_dim}")
    
    # ⚡ MAGIA: Przekształć tokeny → embeddingi
    print(f"\n⚡ Generowanie embeddingów...")
    with torch.no_grad():
        embeddings = embedding_layer(tokens.to(model.device))
    
    # Przenieś na CPU do zapisu
    embeddings_cpu = embeddings.cpu()
    
    print(f"   ✅ Shape: {embeddings_cpu.shape}")  # [1, num_tokens, embedding_dim]
    print(f"   💾 Rozmiar w pamięci: {embeddings_cpu.element_size() * embeddings_cpu.nelement() / (1024**2):.2f} MB")
    
    # Przygotuj metadane
    metadata = {
        "source_file": instructions_file,
        "model_path": model_path,
        "num_tokens": int(num_tokens),
        "embedding_dim": int(embedding_layer.embedding_dim),
        "original_text": instructions_text,
        "shape": list(embeddings_cpu.shape),
        "dtype": str(embeddings_cpu.dtype)
    }
    
    # Zapisz kapsułę
    output_path = pathlib.Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    capsule_path = output_path / "memory_capsule.pt"
    metadata_path = output_path / "memory_capsule_meta.json"
    
    torch.save(embeddings_cpu, capsule_path)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Kapsuła zapisana:")
    print(f"   📦 Embeddingi: {capsule_path}")
    print(f"   📄 Metadane: {metadata_path}")
    print("✨ SUKCES! Kapsuła gotowa do użycia.")
    print("=" * 70)
    print(f"\n💡 Teraz użyj tej kapsuły w main.py z parametrem use_memory_capsule=True")
    
    # Zwolnij pamięć
    del model
    del embedding_layer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return capsule_path, metadata_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generator Kapsuły Pamięci z Embeddingami")
    parser.add_argument("--model", type=str, required=True, help="Ścieżka do modelu HuggingFace (lub adaptera LoRA)")
    parser.add_argument("--instructions", type=str, required=True, help="Plik .txt z instrukcjami")
    parser.add_argument("--output", type=str, required=True, help="Folder wyjściowy")
    parser.add_argument("--4bit", action="store_true", help="Użyj kwantyzacji 4-bit")
    parser.add_argument("--cpu", action="store_true", help="Wymuś CPU zamiast GPU")
    parser.add_argument("--lora", action="store_true", help="Model to adapter LoRA")
    parser.add_argument("--base-model", type=str, help="Ścieżka do base modelu (wymagane dla LoRA)")
    
    args = parser.parse_args()
    
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.lora and not args.base_model:
        print("❌ BŁĄD: --base-model jest wymagane gdy używasz --lora")
        exit(1)
    
    try:
        create_embedding_capsule(
            model_path=args.model,
            instructions_file=args.instructions,
            output_folder=args.output,
            use_4bit=args.__dict__.get('4bit', False),
            device=device,
            is_lora=args.lora,
            base_model_path=args.base_model
        )
    except Exception as e:
        print(f"\n❌ BŁĄD: {e}")
        import traceback
        traceback.print_exc()

