"""
Generator 'Kapsuły Pamięci' - Prosty konwerter tekstu na pre-prompt.

UWAGA: Ponieważ używasz llama-cpp-python (GGUF), nie możemy wstrzykiwać 
embedingów bezpośrednio. Zamiast tego, ten skrypt przygotowuje:
1. Sformatowany pre-prompt do wstawiania na początku każdej konwersacji
2. Opcjonalnie: embeddingi tekstowe przez sentence-transformers (dla semantycznego wyszukiwania)

To jest uproszczona "Pakowalnia" dopasowana do Twojego stacku.
"""

import pathlib
import json

def create_memory_capsule(
    text_file_path: str,
    model_folder: str,
    output_file: str = "memory_capsule.json"
):
    """
    Tworzy 'Kapsułę Pamięci' z pliku tekstowego.
    
    Args:
        text_file_path: Ścieżka do pliku .txt z instrukcjami/kontekstem
        model_folder: Folder gdzie zapisać kapsułę
        output_file: Nazwa pliku wyjściowego (domyślnie: memory_capsule.json)
    """
    
    print(f"📖 Wczytywanie tekstu z: {text_file_path}")
    with open(text_file_path, 'r', encoding='utf-8') as f:
        instruction_text = f.read()
    
    print(f"📝 Długość tekstu: {len(instruction_text)} znaków")
    
    # Przygotuj sformatowany prompt systemowy
    formatted_system_prompt = f"<system>\n{instruction_text}\n</system>\n\n"
    
    # Zapisz jako JSON z dodatkowymi metadanymi
    capsule_data = {
        'original_text': instruction_text,
        'formatted_prompt': formatted_system_prompt,
        'created_at': pathlib.Path(text_file_path).stat().st_mtime,
        'source_file': text_file_path,
        'num_chars': len(instruction_text),
        'num_words': len(instruction_text.split())
    }
    
    output_path = pathlib.Path(model_folder) / output_file
    print(f"💾 Zapisywanie kapsuły do: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(capsule_data, f, ensure_ascii=False, indent=2)
    
    file_size_kb = output_path.stat().st_size / 1024
    print(f"✅ Kapsuła pamięci utworzona!")
    print(f"📁 Rozmiar pliku: {file_size_kb:.2f} KB")
    print(f"📊 Słów: {capsule_data['num_words']}, Znaków: {capsule_data['num_chars']}")
    print(f"🎯 Możesz teraz użyć tego pliku w swoim serwerze LLM")
    
    return output_path

if __name__ == "__main__":
    # Ścieżki - dostosuj do swojego setupu
    MODEL_FOLDER = "qwen-3b-distilled_latest"
    TEXT_FILE = f"{MODEL_FOLDER}/instructions.txt"
    OUTPUT_FILE = "memory_capsule.json"
    
    print("=" * 60)
    print("🧠 GENERATOR KAPSUŁY PAMIĘCI")
    print("=" * 60)
    
    try:
        capsule_path = create_memory_capsule(
            text_file_path=TEXT_FILE,
            model_folder=MODEL_FOLDER,
            output_file=OUTPUT_FILE
        )
        
        print("\n" + "=" * 60)
        print("✨ SUKCES! Kapsuła gotowa do użycia.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ BŁĄD: {e}")
        import traceback
        traceback.print_exc()
