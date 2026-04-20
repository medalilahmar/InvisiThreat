import json
import shutil
from pathlib import Path

def clean_all_caches():
    
    caches_to_clean = [
        "data/ai_scores_cache.json",
        "data/findings_cache.json",
        ".cache",
        "__pycache__",
    ]
    
    print("=" * 70)
    print("🧹 NETTOYAGE COMPLET DES CACHES")
    print("=" * 70)
    
    cleaned = 0
    
    for cache_path in caches_to_clean:
        cache = Path(cache_path)
        
        print(f"\n {cache_path}:")
        
        if cache.is_file():
            try:
                if cache.suffix == ".json":
                    with open(cache, "r", encoding="utf-8") as f:
                        json.load(f)
                    
                    with open(cache, "w", encoding="utf-8") as f:
                        json.dump({}, f, ensure_ascii=False, indent=2)
                    
                    print(f"   ✓ Contenu vidé")
                else:
                    cache.unlink()
                    print(f"   ✓ Fichier supprimé")
                
                cleaned += 1
            
            except json.JSONDecodeError:
                print(f"  Fichier JSON corrompu, suppression...")
                cache.unlink()
                
                cache.parent.mkdir(parents=True, exist_ok=True)
                with open(cache, "w", encoding="utf-8") as f:
                    json.dump({}, f)
                
                print(f"   ✓ Nouveau fichier créé")
                cleaned += 1
            
            except Exception as e:
                print(f"   ❌ Erreur: {e}")
        
        elif cache.is_dir():
            try:
                shutil.rmtree(cache)
                print(f"   ✓ Dossier supprimé")
                cleaned += 1
            except Exception as e:
                print(f"   ❌ Erreur: {e}")
        
        else:
            print(f"   ⚠️ Non trouvé")
    
    print("\n" + "=" * 70)
    print(f"✅ NETTOYAGE TERMINÉ: {cleaned} éléments nettoyés")
    print("=" * 70 + "\n")

def clean_defectdojo_cache():
    """Nettoie le cache spécifique à DefectDojo"""
    
    print("\n" + "=" * 70)
    print("🔄 RAFRAÎCHISSEMENT DU CACHE DEFECTDOJO")
    print("=" * 70)
    
    cache_files = [
        "data/ai_scores_cache.json",
    ]
    
    for cache_file in cache_files:
        cache = Path(cache_file)
        print(f"\n📄 {cache_file}:")
        
        if cache.exists():
            try:
                # Lire avec UTF-8
                with open(cache, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                print(f"   • Entrées actuelles: {len(data)}")
                
                # Vider
                with open(cache, "w", encoding="utf-8") as f:
                    json.dump({}, f, ensure_ascii=False, indent=2)
                
                print(f"   ✓ Cache vidé et reinitalisé")
            
            except Exception as e:
                print(f"   ❌ Erreur: {e}")
                print(f"   Suppression et recréation...")
                
                cache.unlink()
                cache.parent.mkdir(parents=True, exist_ok=True)
                
                with open(cache, "w", encoding="utf-8") as f:
                    json.dump({}, f)
                
                print(f"   ✓ Nouveau cache créé")
        else:
            print(f"   ⚠️ Fichier non trouvé, création...")
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, "w", encoding="utf-8") as f:
                json.dump({}, f)
            print(f"   ✓ Créé")
    
    print("\n" + "=" * 70)
    print("✅ CACHE DEFECTDOJO RAFRAÎCHI")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    clean_all_caches()
    clean_defectdojo_cache()
    
    print("\n💡 Prochaines étapes:")
    print("   1. Redémarrez l'API: python api.py")
    print("   2. Visitez: http://localhost:8081/docs")
    print("   3. Testez les endpoints DefectDojo")
    print("\n")