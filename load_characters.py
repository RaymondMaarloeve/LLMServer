"""
Skrypt do ładowania przykładowych postaci do bazy wektorowej.
Używa API /add-character-context do dodania postaci z pliku example_characters.json
"""
import requests
import json
import sys

# Konfiguracja
SERVER_URL = "http://localhost:5000"
CHARACTERS_FILE = "example_characters.json"

def load_characters():
    """Wczytaj i dodaj postaci do serwera"""
    
    # Wczytaj plik z postaciami
    try:
        with open(CHARACTERS_FILE, 'r', encoding='utf-8') as f:
            characters = json.load(f)
    except FileNotFoundError:
        print(f"❌ Nie znaleziono pliku {CHARACTERS_FILE}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Błąd parsowania JSON: {e}")
        return False
    
    print(f"📖 Wczytano {len(characters)} postaci z pliku")
    print()
    
    # Dodaj każdą postać do serwera
    success_count = 0
    for char in characters:
        try:
            response = requests.post(
                f"{SERVER_URL}/add-character-context",
                json=char,
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"✅ Dodano: {char['name']} ({char['character_id']})")
                success_count += 1
            else:
                print(f"❌ Błąd dla {char['name']}: {response.json().get('message', 'Unknown error')}")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Nie można połączyć się z serwerem na {SERVER_URL}")
            print("   Upewnij się, że serwer jest uruchomiony (python main.py)")
            return False
        except Exception as e:
            print(f"❌ Błąd dla {char['name']}: {e}")
    
    print()
    print(f"🎉 Pomyślnie dodano {success_count}/{len(characters)} postaci")
    return True

def list_characters():
    """Wyświetl wszystkie postaci w bazie"""
    try:
        response = requests.get(f"{SERVER_URL}/list-characters", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            characters = data.get('characters', [])
            
            print(f"\n📋 Postaci w bazie ({data.get('count', 0)}):")
            print("=" * 60)
            
            for char in characters:
                print(f"\n🎭 {char.get('name', char['character_id'])}")
                print(f"   ID: {char['character_id']}")
                print(f"   Kontekst: {char['context'][:100]}...")
        else:
            print(f"❌ Błąd: {response.json().get('message', 'Unknown error')}")
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Nie można połączyć się z serwerem na {SERVER_URL}")
    except Exception as e:
        print(f"❌ Błąd: {e}")

def test_query():
    """Przetestuj wyszukiwanie semantyczne"""
    test_queries = [
        "kto robi broń",
        "chcę się wyspowiadać",
        "gdzie mogę napić się piwa",
        "potrzebuję leku na ranę"
    ]
    
    print("\n🔍 Test wyszukiwania semantycznego:")
    print("=" * 60)
    
    for query in test_queries:
        try:
            response = requests.post(
                f"{SERVER_URL}/get-character-context",
                json={"query": query, "n_results": 1},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['results']:
                    result = data['results'][0]
                    print(f"\n❓ '{query}'")
                    print(f"   → {result.get('name', result['character_id'])}")
                    print(f"      (Dystans: {result.get('distance', 'N/A')})")
            
        except Exception as e:
            print(f"❌ Błąd dla zapytania '{query}': {e}")

if __name__ == "__main__":
    print("🎮 Raymond Maarloeve - Loader postaci")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "list":
            list_characters()
        elif command == "test":
            test_query()
        else:
            print(f"❌ Nieznana komenda: {command}")
            print("\nDostępne komendy:")
            print("  python load_characters.py        - wczytaj postaci z pliku")
            print("  python load_characters.py list   - wyświetl postaci w bazie")
            print("  python load_characters.py test   - przetestuj wyszukiwanie")
    else:
        # Domyślnie - wczytaj postaci
        if load_characters():
            print("\n💡 Możesz teraz użyć:")
            print("   python load_characters.py list  - aby zobaczyć postaci")
            print("   python load_characters.py test  - aby przetestować wyszukiwanie")
