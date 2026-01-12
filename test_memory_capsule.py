"""
Prosty skrypt testowy do sprawdzenia działania Kapsuły Pamięci
"""

import requests
import json

SERVER_URL = "http://localhost:5000"

def test_memory_capsule():
    print("=" * 60)
    print("🧪 TEST KAPSUŁY PAMIĘCI")
    print("=" * 60)
    
    # 1. Zarejestruj model
    print("\n1️⃣ Rejestrowanie modelu...")
    response = requests.post(f"{SERVER_URL}/register", json={
        "model_id": "qwen-detective",
        "model_path": "qwen-3b-distilled_latest/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    })
    print(f"   Status: {response.status_code}")
    print(f"   Odpowiedź: {response.json()}")
    
    # 2. Załaduj model (automatycznie załaduje kapsułę)
    print("\n2️⃣ Ładowanie modelu (z automatycznym wykryciem kapsuły)...")
    response = requests.post(f"{SERVER_URL}/load", json={
        "model_id": "qwen-detective",
        "model_path": "qwen-3b-distilled_latest/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "n_ctx": 2048,
        "n_gpu_layers": -1
    })
    print(f"   Status: {response.status_code}")
    result = response.json()
    print(f"   Model załadowany: {result.get('success')}")
    print(f"   Ma kapsułę: {result.get('has_memory_capsule')}")
    
    if not result.get('has_memory_capsule'):
        print("\n⚠️ UWAGA: Kapsuła pamięci nie została znaleziona!")
        print("   Uruchom: python create_memory_capsule.py")
        return
    
    # 3. Test chat BEZ wyraźnego system message
    print("\n3️⃣ Test chat (kapsuła powinna być automatycznie wstrzyknięta)...")
    response = requests.post(f"{SERVER_URL}/chat", json={
        "model_id": "qwen-detective",
        "messages": [
            {
                "role": "user",
                "content": "Dzień dobry! Jestem detektywem przysłanym z miasta. Muszę zadać panu kilka pytań o wczorajszy wieczór."
            }
        ],
        "max_tokens": 200,
        "temperature": 0.8,
        "use_memory_capsule": True
    })
    
    print(f"   Status: {response.status_code}")
    result = response.json()
    
    if result.get('success'):
        print(f"\n   💬 Odpowiedź NPC:")
        print(f"   {result['response']}")
        print(f"\n   📊 Statystyki:")
        print(f"   - Tokeny: {result.get('total_tokens')}")
        print(f"   - Czas: {result.get('generation_time')}s")
    else:
        print(f"   ❌ Błąd: {result.get('message')}")
    
    # 4. Test z wyłączoną kapsułą (dla porównania)
    print("\n4️⃣ Test chat z WYŁĄCZONĄ kapsułą (dla porównania)...")
    response = requests.post(f"{SERVER_URL}/chat", json={
        "model_id": "qwen-detective",
        "messages": [
            {
                "role": "user",
                "content": "Dzień dobry! Jestem detektywem. Co pan robił wczoraj?"
            }
        ],
        "max_tokens": 100,
        "temperature": 0.8,
        "use_memory_capsule": False  # Wyłączone!
    })
    
    print(f"   Status: {response.status_code}")
    result = response.json()
    
    if result.get('success'):
        print(f"\n   💬 Odpowiedź (bez kontekstu):")
        print(f"   {result['response']}")
        print(f"\n   ℹ️ Zauważ różnicę - model nie wie że jest NPC w grze!")
    
    print("\n" + "=" * 60)
    print("✅ Test zakończony!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_memory_capsule()
    except requests.exceptions.ConnectionError:
        print("\n❌ BŁĄD: Nie można połączyć się z serwerem!")
        print("   Upewnij się że serwer jest uruchomiony: python main.py")
    except Exception as e:
        print(f"\n❌ BŁĄD: {e}")
        import traceback
        traceback.print_exc()
