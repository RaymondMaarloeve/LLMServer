# 🚀 Quick Start - Embeddingi w 5 minut

## 1. Zainstaluj (Windows)

```powershell
.\setup.ps1
```

Lub ręcznie:
```bash
pip install -e .
pip install llama-cpp-python
```

## 2. Uruchom serwer

```bash
python main.py
```

Serwer: `http://localhost:5000`

## 3. Wczytaj przykładowe postacie

```bash
python load_characters.py
```

To doda 5 postaci:
- 🔨 Jan Kowal (kowal)
- ⛪ Ojciec Tomasz (ksiądz)
- 🍺 Anna Karczmarka (karczmarka)
- 🛡️ Piotr Strażnik (strażnik)
- 🌿 Marcin Zielarz (znachor)

## 4. Przetestuj

```bash
python load_characters.py test
```

## 5. Użyj w grze

### A) Rozmowa z konkretną postacią:

```json
POST http://localhost:5000/chat
{
  "model_id": "twoj_model",
  "character_id": "kowal_jan",
  "messages": [
    {"role": "user", "content": "Witaj! Potrzebuję miecza."}
  ]
}
```

### B) Automatyczny dobór postaci:

```json
POST http://localhost:5000/chat
{
  "model_id": "twoj_model",
  "use_embedding": true,
  "messages": [
    {"role": "user", "content": "Kto może naprawić broń?"}
  ]
}
```

System sam wybierze kowala! 🎯

## Dodawanie własnych postaci

### Przez skrypt:

Edytuj `example_characters.json` i:
```bash
python load_characters.py
```

### Przez API:

```bash
curl -X POST http://localhost:5000/add-character-context \
  -H "Content-Type: application/json" \
  -d '{
    "character_id": "twoja_postac",
    "name": "Imię Postaci",
    "context": "Pełny opis postaci jako system prompt..."
  }'
```

## Więcej info

- 📖 Pełna dokumentacja: `EMBEDDINGS.md`
- 🔗 Przykłady HTTP: `LLMServer_embeddings.http`
- 🎮 Główne repo: https://github.com/RaymondMaarloeve/RaymondMaarloeve

---

**To wszystko! System gotowy do użycia! 🎉**
