"""
Test serwera z kapsułą embeddingów - uproszczony skrypt
"""
import requests
import json
import time

SERVER_URL = "http://localhost:5000"

def test_server():
    print("=" * 80)
    print("🧪 TEST SERWERA Z KAPSUŁĄ EMBEDDINGÓW")
    print("=" * 80)
    
    # 1. Sprawdź status serwera
    print("\n1️⃣ Sprawdzanie statusu serwera...")
    try:
        response = requests.get(f"{SERVER_URL}/status", timeout=5)
        print(f"   ✅ Serwer działa! Status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Błąd połączenia: {e}")
        return
    
    # 2. Załaduj model z LoRA
    print("\n2️⃣ Ładowanie modelu Qwen z LoRA adapterem i kapsułą embeddingów...")
    print("   (To może potrwać kilka minut - pobieranie bazowego modelu z HuggingFace)")
    
    load_config = {
        "model_id": "qwen-detective",
        "model_path": "qwen-3b-distilled_latest",
        "is_lora": True,
        "base_model_path": "Qwen/Qwen2.5-3B-Instruct",
        "merge_adapter": True,
        "device": "cuda",
        "torch_dtype": "float16",
        "max_length": 2048
    }
    
    try:
        response = requests.post(f"{SERVER_URL}/load", json=load_config, timeout=600)
        result = response.json()
        print(f"   Status: {response.status_code}")
        print(f"   Odpowiedź: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        if not result.get('success'):
            print(f"\n   ❌ Nie udało się załadować modelu!")
            return
            
        if result.get('has_memory_capsule'):
            print(f"\n   ✅ Model załadowany z kapsułą embeddingów!")
        else:
            print(f"\n   ⚠️ Model załadowany, ale BEZ kapsuły embeddingów")
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Błąd podczas ładowania: {e}")
        return
    
    # 3. Test chat Z kapsułą
    print("\n3️⃣ Test chat Z kapsułą embeddingów...")
    print("   Pytanie: 'Dzień dobry! Jestem detektywem. Co pan robił wczoraj wieczorem?'")
    
    chat_request = {
        "model_id": "qwen-detective",
        "messages": [
            {
                "role": "user",
                "content": "Dzień dobry! Jestem detektywem przysłanym z miasta. Co pan robił wczoraj wieczorem?"
            }
        ],
        "max_tokens": 150,
        "temperature": 0.8,
        "use_memory_capsule": True
    }
    
    try:
        print("   Generowanie odpowiedzi...")
        response = requests.post(f"{SERVER_URL}/chat", json=chat_request, timeout=120)
        result = response.json()
        
        if result.get('success'):
            print(f"\n   💬 Odpowiedź NPC (Z KAPSUŁĄ):")
            print(f"   {'-' * 70}")
            print(f"   {result['response']}")
            print(f"   {'-' * 70}")
            print(f"\n   📊 Statystyki:")
            print(f"   - Użyto kapsuły: {result.get('used_embedding_capsule')}")
            print(f"   - Czas generowania: {result.get('generation_time')}s")
        else:
            print(f"   ❌ Błąd: {result.get('message')}")
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Błąd podczas chatu: {e}")
        return
    
    # 4. Test chat BEZ kapsuły (dla porównania)
    print("\n4️⃣ Test chat BEZ kapsuły (dla porównania)...")
    print("   To samo pytanie, ale bez kontekstu NPC")
    
    chat_request_no_capsule = {
        "model_id": "qwen-detective",
        "messages": [
            {
                "role": "user",
                "content": "Dzień dobry! Jestem detektywem przysłanym z miasta. Co pan robił wczoraj wieczorem?"
            }
        ],
        "max_tokens": 150,
        "temperature": 0.8,
        "use_memory_capsule": False
    }
    
    try:
        print("   Generowanie odpowiedzi...")
        time.sleep(1)
        response = requests.post(f"{SERVER_URL}/chat", json=chat_request_no_capsule, timeout=120)
        result = response.json()
        
        if result.get('success'):
            print(f"\n   💬 Odpowiedź (BEZ KAPSUŁY):")
            print(f"   {'-' * 70}")
            print(f"   {result['response']}")
            print(f"   {'-' * 70}")
            print(f"\n   📊 Statystyki:")
            print(f"   - Użyto kapsuły: {result.get('used_embedding_capsule')}")
            print(f"   - Czas generowania: {result.get('generation_time')}s")
            print(f"\n   ℹ️ Porównaj różnicę - bez kapsuły model nie wie że jest NPC w grze!")
        else:
            print(f"   ❌ Błąd: {result.get('message')}")
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Błąd podczas chatu: {e}")
    
    print("\n" + "=" * 80)
    print("✅ TEST ZAKOŃCZONY!")
    print("=" * 80)
    print("\nPodsumowanie:")
    print("- Kapsuła embeddingów wstrzykuje kontekst systemowy bezpośrednio jako embeddingi")
    print("- Model 'pamięta' że jest NPC w średniowiecznej grze detektywistycznej")
    print("- Nie trzeba za każdym razem przesyłać długiego system message")
    print("=" * 80)

if __name__ == "__main__":
    test_server()
