# 🧠 Kapsuła Pamięci - Instrukcja Użycia

System "Kapsuły Pamięci" pozwala na automatyczne wstrzykiwanie kontekstu systemowego do każdego zapytania bez konieczności przesyłania go za każdym razem.

## 📋 Kroki użycia:

### 1. Przygotuj instrukcje
Plik `qwen-3b-distilled_latest/instructions.txt` już zawiera instrukcje dla NPC w grze detektywistycznej.

### 2. Wygeneruj kapsułę pamięci
```powershell
python create_memory_capsule.py
```

To utworzy plik `qwen-3b-distilled_latest/memory_capsule.json` z przygotowanym kontekstem.

### 3. Uruchom serwer
```powershell
python main.py
```

### 4. Załaduj model
Wyślij POST request do `/load`:
```json
{
    "model_id": "qwen-detective",
    "model_path": "qwen-3b-distilled_latest/model.gguf",
    "n_ctx": 2048,
    "n_gpu_layers": -1
}
```

**Serwer automatycznie wykryje i wczyta `memory_capsule.json` z folderu modelu!**

### 5. Testuj chat
Wyślij POST request do `/chat`:
```json
{
    "model_id": "qwen-detective",
    "messages": [
        {
            "role": "user",
            "content": "Dzień dobry, jestem detektywem. Muszę zadać panu kilka pytań o wczorajszy wieczór."
        }
    ],
    "max_tokens": 200,
    "temperature": 0.8
}
```

**Kapsuła pamięci zostanie automatycznie wstrzyknięta jako system message!**

## 🎛️ Opcje sterowania

### Wyłączenie kapsuły dla pojedynczego zapytania:
```json
{
    "model_id": "qwen-detective",
    "use_memory_capsule": false,
    "messages": [...]
}
```

### Sprawdzenie czy model ma kapsułę:
Odpowiedź z `/load` zawiera pole `has_memory_capsule: true/false`

## 🔧 Jak to działa?

1. **Pakowalnia** (raz):
   - `create_memory_capsule.py` czyta `instructions.txt`
   - Zapisuje jako `memory_capsule.json` w folderze modelu

2. **Wczytywanie** (przy ładowaniu modelu):
   - `load_memory_capsule()` szuka `memory_capsule.json` w folderze modelu
   - Wczytuje do globalnej zmiennej `memory_capsules[model_id]`

3. **Wstrzykiwanie** (przy każdym /chat):
   - Jeśli `use_memory_capsule=true` (domyślnie)
   - I model ma załadowaną kapsułę
   - I messages nie ma jeszcze system message
   - → Dodaje kapsułę jako pierwszy message z role="system"

## 💡 Zalety

✅ Jeden raz przetwarzasz instrukcje  
✅ Przy każdym zapytaniu są automatycznie dodawane  
✅ Nie musisz przesyłać długiego tekstu w każdym requeście  
✅ Możesz mieć różne kapsuły dla różnych modeli  
✅ Łatwo wyłączyć (use_memory_capsule=false)  

## 📝 Modyfikacja instrukcji

Jeśli chcesz zmienić instrukcje:
1. Edytuj `qwen-3b-distilled_latest/instructions.txt`
2. Uruchom ponownie `python create_memory_capsule.py`
3. Zrestartuj serwer lub przeładuj model

## 🎮 Przykład użycia w grze

```python
import requests

# Załaduj model z kapsułą
response = requests.post("http://localhost:5000/load", json={
    "model_id": "npc-blacksmith",
    "model_path": "qwen-3b-distilled_latest/model.gguf"
})
print(f"Kapsuła: {response.json()['has_memory_capsule']}")

# Rozmawiaj z NPC
response = requests.post("http://localhost:5000/chat", json={
    "model_id": "npc-blacksmith",
    "messages": [
        {"role": "user", "content": "Co pan robił wczoraj wieczorem?"}
    ],
    "max_tokens": 150
})

print(f"NPC: {response.json()['response']}")
```

## 🔍 Debugowanie

Gdy kapsuła jest wstrzykiwana, w konsoli serwera zobaczysz:
```
🧠 Wczytywanie kapsuły pamięci dla modelu 'qwen-detective' z: ...
✅ Kapsuła pamięci wczytana: 123 słów
🧠 Wstrzyknięto kapsułę pamięci do konwersacji (1234 znaków)
```

Jeśli brak kapsuły:
```
ℹ️ Brak kapsuły pamięci dla modelu 'qwen-detective' w: ...
```
