"""
Prosty skrypt do testowania systemu embeddingów bez uruchamiania pełnego serwera.
Testuje tylko ChromaDB i embeddingi (bez LLM).
"""
import chromadb
from chromadb.utils import embedding_functions

def test_embeddings():
    print("🧪 Test systemu embeddingów")
    print("=" * 60)
    
    # 1. Inicjalizacja
    print("\n1️⃣ Inicjalizacja ChromaDB...")
    client = chromadb.Client()
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    collection = client.get_or_create_collection(
        name="test_characters",
        embedding_function=embedding_fn
    )
    print("   ✅ ChromaDB zainicjalizowane")
    
    # 2. Dodanie testowych postaci
    print("\n2️⃣ Dodawanie testowych postaci...")
    
    characters = [
        {
            "id": "blacksmith",
            "text": "Jesteś kowalem, który robi miecze, zbroje i narzędzia. Pracujesz w kuźni.",
            "name": "Jan"
        },
        {
            "id": "priest",
            "text": "Jesteś księdzem w kościele. Odprawiasz msze i spowiadasz ludzi.",
            "name": "Tomasz"
        },
        {
            "id": "innkeeper",
            "text": "Jesteś karczmarzem. Prowadzisz karczmę gdzie ludzie piją piwo i słuchają plotek.",
            "name": "Anna"
        }
    ]
    
    for char in characters:
        collection.add(
            documents=[char["text"]],
            metadatas=[{"name": char["name"]}],
            ids=[char["id"]]
        )
        print(f"   ✅ Dodano: {char['name']}")
    
    # 3. Test wyszukiwania po ID
    print("\n3️⃣ Test wyszukiwania po ID...")
    result = collection.get(ids=["blacksmith"])
    print(f"   Znaleziono: {result['metadatas'][0]['name']}")
    print(f"   Kontekst: {result['documents'][0][:50]}...")
    
    # 4. Test wyszukiwania semantycznego
    print("\n4️⃣ Test wyszukiwania semantycznego...")
    
    queries = [
        "potrzebuję naprawić mój miecz",
        "chcę się wyspowiadać",
        "gdzie mogę napić się piwa"
    ]
    
    for query in queries:
        results = collection.query(
            query_texts=[query],
            n_results=1
        )
        
        if results['documents'] and results['documents'][0]:
            char_name = results['metadatas'][0][0]['name']
            distance = results['distances'][0][0] if 'distances' in results else 0
            
            print(f"\n   ❓ '{query}'")
            print(f"   ✅ Znaleziono: {char_name}")
            print(f"      Dystans: {distance:.4f} (im mniejszy tym lepiej)")
    
    # 5. Test wszystkich postaci
    print("\n5️⃣ Lista wszystkich postaci...")
    all_chars = collection.get()
    print(f"   Liczba postaci w bazie: {len(all_chars['ids'])}")
    for i, char_id in enumerate(all_chars['ids']):
        name = all_chars['metadatas'][i]['name']
        print(f"   - {name}")
    
    print("\n" + "=" * 60)
    print("🎉 Test zakończony pomyślnie!")
    print("\n💡 System embeddingów działa poprawnie.")
    print("   Możesz teraz uruchomić pełny serwer: python main.py")

if __name__ == "__main__":
    try:
        test_embeddings()
    except ImportError as e:
        print(f"❌ Błąd importu: {e}")
        print("\n💡 Zainstaluj zależności:")
        print("   pip install chromadb sentence-transformers")
    except Exception as e:
        print(f"❌ Błąd: {e}")
        import traceback
        traceback.print_exc()
