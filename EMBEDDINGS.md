# 🧠 System Embeddingów dla NPC - Instrukcja

## Co to jest?

System embeddingów w LLMServer pozwala na automatyczne zarządzanie kontekstami postaci (NPC) w grze. Zamiast ręcznie podawać "system prompt" dla każdej postaci przy każdym requeście, system:

1. **Przechowuje** opisy postaci w bazie wektorowej (ChromaDB)
2. **Generuje embeddingi** (wektory semantyczne) dla każdego opisu
3. **Automatycznie znajduje** najlepiej pasującą postać na podstawie kontekstu rozmowy
4. **Wstrzykuje** odpowiedni kontekst do LLM przed generowaniem odpowiedzi

## Jak to działa?

### 1. Dodawanie postaci

Każda postać ma:
- **character_id**: unikalny identyfikator (np. "kowal_jan")
- **name**: imię postaci
- **context**: pełny opis postaci - kim jest, jak mówi, czym się zajmuje

```json
{
  "character_id": "kowal_jan",
  "name": "Jan Kowal",
  "context": "Jesteś Janem Kowalem, doświadczonym kowalem..."
}
```

### 2. Wyszukiwanie postaci

System może znaleźć postać na dwa sposoby:

**A) Bezpośrednio po ID:**
```json
POST /chat
{
  "character_id": "kowal_jan",
  "messages": [...]
}
```

**B) Semantycznie (automatycznie):**
```json
POST /chat
{
  "use_embedding": true,
  "messages": [
    {"role": "user", "content": "Kto może naprawić mój miecz?"}
  ]
}
```

System sam znajdzie, że to pytanie pasuje do kowala!

### 3. Automatyczne wstrzykiwanie kontekstu

Gdy znajdzie odpowiednią postać, system automatycznie dodaje jej kontekst jako `system message` na początku konwersacji. LLM dostaje więc pełny kontekst postaci i odpowiada "w charakterze".

## Instalacja i konfiguracja

### 1. Zainstaluj zależności

```bash
# Automatycznie (Windows)
.\setup.ps1

# Lub ręcznie
pip install -e .
pip install llama-cpp-python
```

### 2. Uruchom serwer

```bash
python main.py
```

Serwer wystartuje na `http://localhost:5000`

### 3. Dodaj przykładowe postacie

```bash
# Wczytaj 5 przykładowych postaci z pliku
python load_characters.py

# Sprawdź co zostało dodane
python load_characters.py list

# Przetestuj wyszukiwanie semantyczne
python load_characters.py test
```

## Przykłady użycia

### Dodanie nowej postaci przez API

```bash
curl -X POST http://localhost:5000/add-character-context \
  -H "Content-Type: application/json" \
  -d '{
    "character_id": "straznik_piotr",
    "name": "Piotr Strażnik",
    "context": "Jesteś Piotrem, kapitanem straży..."
  }'
```

### Rozmowa z konkretną postacią

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "moj_model",
    "character_id": "kowal_jan",
    "messages": [
      {"role": "user", "content": "Dzień dobry! Co dziś porabiasz?"}
    ]
  }'
```

### Rozmowa z automatycznym doborem postaci

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "moj_model",
    "use_embedding": true,
    "messages": [
      {"role": "user", "content": "Potrzebuję naprawić zbroję"}
    ]
  }'
```

System automatycznie wybierze kowala na podstawie treści wiadomości!

## Pełna lista endpointów

### Zarządzanie postaciami

| Endpoint | Metoda | Opis |
|----------|--------|------|
| `/add-character-context` | POST | Dodaj/zaktualizuj postać |
| `/get-character-context` | POST | Pobierz postać po ID lub wyszukaj semantycznie |
| `/list-characters` | GET | Lista wszystkich postaci |

### Standardowe endpointy (bez zmian)

| Endpoint | Metoda | Opis |
|----------|--------|------|
| `/load` | POST | Załaduj model LLM |
| `/chat` | POST | Chat z LLM (teraz z embeddingami!) |
| `/unload` | POST | Wyładuj model |
| `/status` | GET | Status serwera i modeli |
| `/register` | POST | Zarejestruj model |

## Pliki w projekcie

```
LLMServer/
├── main.py                      # Główny serwer (zmodyfikowany z embeddingami)
├── pyproject.toml               # Zależności (dodano chromadb, sentence-transformers)
├── example_characters.json      # 5 przykładowych postaci PL
├── load_characters.py           # Skrypt do ładowania postaci
├── setup.ps1                    # Instalator dla Windows
├── LLMServer_embeddings.http    # Przykłady requestów HTTP
└── EMBEDDINGS.md                # Ta instrukcja
```

## Jak to rozbudować?

### 1. Dodaj więcej postaci

Edytuj `example_characters.json` i dodaj nowe postaci, potem:

```bash
python load_characters.py
```

### 2. Zmień model embeddingów

W `main.py` zmień:

```python
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"  # <- tutaj zmień model
)
```

Modele do wyboru:
- `all-MiniLM-L6-v2` - szybki, lekki (domyślnie)
- `paraphrase-multilingual-MiniLM-L12-v2` - lepszy dla polskiego
- `all-mpnet-base-v2` - lepsze embeddingi, ale wolniejszy

### 3. Zapisz bazę na dysku

Domyślnie ChromaDB działa w pamięci. Aby zapisać na dysku:

```python
chroma_client = chromadb.PersistentClient(path="./chroma_db")
```

## Rozwiązywanie problemów

### Błąd: "Character context system is not initialized"

ChromaDB nie został poprawnie zainicjalizowany. Sprawdź:
1. Czy zainstalowano `chromadb` i `sentence-transformers`
2. Czy są błędy przy starcie serwera
3. Czy model embeddings się pobrał (pierwsze uruchomienie pobiera model)

### Wyszukiwanie nie działa dobrze

1. Używaj modelu lepszego dla polskiego: `paraphrase-multilingual-MiniLM-L12-v2`
2. Pisz bardziej szczegółowe konteksty postaci
3. Zwiększ `n_results` w zapytaniu, żeby zobaczyć więcej wyników

### Wolne działanie

1. Model embeddings pobiera się tylko przy pierwszym uruchomieniu
2. Pierwsze zapytanie może być wolniejsze (inicjalizacja)
3. Użyj lżejszego modelu embeddingów
4. Dla dużej liczby postaci rozważ persistent ChromaDB

## FAQ

**Q: Czy muszę używać embeddingów?**  
A: Nie! Stary sposób (ręczne podawanie system message) nadal działa. Embeddingi są opcjonalne.

**Q: Ile postaci mogę mieć?**  
A: Praktycznie bez limitu. ChromaDB obsługuje miliony wektorów.

**Q: Czy działa offline?**  
A: Tak! Po pierwszym pobraniu modelu wszystko działa lokalnie.

**Q: Czy to spowalnia serwer?**  
A: Minimalnie. Wyszukiwanie w ChromaDB jest bardzo szybkie.

**Q: Czy mogę mieć wiele baz postaci?**  
A: Tak, możesz stworzyć wiele kolekcji w ChromaDB.

## Licencja i kontakt

To jest część projektu **Raymond Maarloeve** - średniowiecznej gry detektywistycznej.

🔗 Główne repo: https://github.com/RaymondMaarloeve/RaymondMaarloeve  
📖 Dokumentacja: https://raymondmaarloeve.github.io/LLMServer/

---

**Powodzenia z Twoją grą! 🎮⚔️**
