"""
DIAGNOSTIC COMPLET: Hiérarchie Product → Engagement → Test → Finding
Détecte pourquoi seulement 42 findings s'affichent au lieu de 500
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

# Import depuis api.py
sys.path.insert(0, str(Path(__file__).parent / "risk-scoring-engine" / "src"))

from dotenv import load_dotenv
load_dotenv()

try:
    from api import DefectDojoClient, HierarchyManager
except ImportError:
    print("❌ Impossible d'importer api.py. Assurez-vous que DEFECTDOJO_API_KEY est défini.")
    sys.exit(1)


def diagnose_hierarchy():
    """Diagnostic complet de la hiérarchie DefectDojo"""
    
    print("\n" + "="*80)
    print("🔍 DIAGNOSTIC: HIÉRARCHIE PRODUCT → ENGAGEMENT → TEST → FINDING")
    print("="*80)
    
    # Initialiser le client
    client = DefectDojoClient(
        os.getenv("DEFECTDOJO_URL", "http://192.168.11.170:8080"),
        os.getenv("DEFECTDOJO_API_KEY", "a8506b7874b044ed31f8d6b847ca4d6b15bdb868"),
    )
    
    # Récupérer les données
    print("\n📥 Récupération des données depuis DefectDojo...")
    products = client.get_products()
    engagements = client.get_engagements()
    tests = client.get_tests()
    findings = client.get_findings(limit=2000 , include_inactive=True)  
    
    print(f"   ✓ {len(products)} produits")
    print(f"   ✓ {len(engagements)} engagements")
    print(f"   ✓ {len(tests)} tests")
    print(f"   ✓ {len(findings)} findings")
    
    # ── SECTION 1: PRODUITS ─────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("1️⃣  PRODUITS")
    print("="*80)
    
    for p in products:
        print(f"\n   ID={p['id']:2d} | {p.get('name', 'N/A'):30s}")
    
    # ── SECTION 2: ENGAGEMENTS ──────────────────────────────────────────────────
    print("\n" + "="*80)
    print("2️⃣  ENGAGEMENTS")
    print("="*80)
    
    print("\n   ID | Nom                            | Produit")
    print("   " + "-"*70)
    for e in engagements:
        prod_name = next((p.get('name', 'N/A') for p in products if p['id'] == e.get('product')), "?")
        print(f"   {e['id']:2d} | {e.get('name', 'N/A'):30s} | {e.get('product'):2d} ({prod_name})")
    
    # ── SECTION 3: TESTS ────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("3️⃣  TESTS → ENGAGEMENTS")
    print("="*80)
    
    tests_by_engagement = defaultdict(list)
    tests_orphaned = 0
    
    for t in tests:
        eng_id = t.get('engagement')
        if eng_id is None:
            tests_orphaned += 1
        else:
            tests_by_engagement[eng_id].append(t['id'])
    
    print(f"\n   Total: {len(tests)} tests")
    print(f"   Orphelins (sans engagement): {tests_orphaned}")
    
    print("\n   Engagement | Nom           | Produit | # Tests")
    print("   " + "-"*70)
    for eng_id in sorted(tests_by_engagement.keys()):
        eng = next((e for e in engagements if e['id'] == eng_id), {})
        prod_id = eng.get('product', '?')
        prod_name = next((p.get('name', '?') for p in products if p['id'] == prod_id), "?")
        num_tests = len(tests_by_engagement[eng_id])
        print(f"   {eng_id:10d} | {eng.get('name', 'N/A'):13s} | {prod_id:7} | {num_tests:8d}")
    
    # ── SECTION 4: FINDINGS RAW ─────────────────────────────────────────────────
    print("\n" + "="*80)
    print("4️⃣  FINDINGS BRUTS")
    print("="*80)
    
    findings_with_test = sum(1 for f in findings if f.get('test') is not None)
    findings_without_test = len(findings) - findings_with_test
    
    findings_by_engagement = defaultdict(list)
    findings_by_product = defaultdict(list)
    findings_by_test = defaultdict(list)
    
    for f in findings:
        eng_id = f.get('engagement')
        prod_id = f.get('product')
        test_id = f.get('test')
        
        if eng_id:
            findings_by_engagement[eng_id].append(f['id'])
        if prod_id:
            findings_by_product[prod_id].append(f['id'])
        if test_id:
            findings_by_test[test_id].append(f['id'])
    
    print(f"\n   Total: {len(findings)} findings")
    print(f"   Avec test_id valide: {findings_with_test}")
    print(f"   SANS test_id (orphelins): {findings_without_test} ⚠️")
    
    print("\n   Par PRODUCT:")
    print("   Produit | Nom                    | # Findings")
    print("   " + "-"*70)
    for prod_id in sorted(findings_by_product.keys()):
        prod = next((p for p in products if p['id'] == prod_id), {})
        num_findings = len(findings_by_product[prod_id])
        print(f"   {prod_id:7} | {prod.get('name', 'N/A'):22s} | {num_findings:11d}")
    
    print("\n   Par ENGAGEMENT:")
    print("   Engagement | Nom           | Product | # Findings")
    print("   " + "-"*70)
    for eng_id in sorted(findings_by_engagement.keys()):
        eng = next((e for e in engagements if e['id'] == eng_id), {})
        prod_id = eng.get('product', '?')
        num_findings = len(findings_by_engagement[eng_id])
        print(f"   {eng_id:10} | {eng.get('name', 'N/A'):13s} | {prod_id:7} | {num_findings:11d}")
    
    # ── SECTION 5: ENRICHISSEMENT ───────────────────────────────────────────────
    print("\n" + "="*80)
    print("5️⃣  ENRICHISSEMENT VIA HIERARCHYMANAGER")
    print("="*80)
    
    hierarchy = HierarchyManager(client)
    enriched = hierarchy.enrich_findings(findings)
    
    resolved = sum(1 for f in enriched if f.get('product_id') is not None)
    unresolved = len(enriched) - resolved
    
    print(f"\n   Findings enrichis: {resolved}/{len(enriched)}")
    print(f"   Findings résolus: {resolved} ✓")
    print(f"   Findings NON résolus (orphelins): {unresolved} ⚠️")
    
    if unresolved > 0:
        print(f"\n   ⚠️ EXEMPLES D'ORPHELINS:")
        orphans = [f for f in enriched if f.get('product_id') is None][:5]
        for o in orphans:
            print(f"      Finding {o['id']:4d} | test={str(o.get('test')):6s} | engagement={str(o.get('engagement')):6s} | product={o.get('product')}")
    
    # ── SECTION 6: DISTRIBUTION PAR PRODUIT APRÈS ENRICHISSEMENT ────────────────
    print("\n" + "="*80)
    print("6️⃣  DISTRIBUTION APRÈS ENRICHISSEMENT")
    print("="*80)
    
    enriched_by_product = defaultdict(list)
    for f in enriched:
        prod_id = f.get('product_id')
        if prod_id is not None:
            enriched_by_product[prod_id].append(f['id'])
    
    print("\n   Produit | Nom                    | # Findings (enrichis)")
    print("   " + "-"*70)
    for prod_id in sorted(enriched_by_product.keys()):
        prod = next((p for p in products if p['id'] == prod_id), {})
        num_findings = len(enriched_by_product[prod_id])
        print(f"   {prod_id:7} | {prod.get('name', 'N/A'):22s} | {num_findings:19d}")
    
    # ── SECTION 7: FOCUS PRODUCT 5 ──────────────────────────────────────────────
    print("\n" + "="*80)
    print("7️⃣  FOCUS: PRODUCT 5 (InvisiThreat)")
    print("="*80)
    
    product_5 = next((p for p in products if p['id'] == 5), None)
    if not product_5:
        print("   ❌ Product 5 non trouvé!")
    else:
        print(f"\n   Produit: {product_5['name']}")
        
        # Engagements du product 5
        engs_prod_5 = [e for e in engagements if e.get('product') == 5]
        print(f"\n   Engagements: {len(engs_prod_5)}")
        for e in engs_prod_5:
            print(f"      - ID={e['id']:2d}: {e['name']}")
        
        # Tests du product 5
        tests_prod_5 = []
        for t in tests:
            eng_id = t.get('engagement')
            if eng_id and any(e['id'] == eng_id for e in engs_prod_5):
                tests_prod_5.append(t['id'])
        
        print(f"\n   Tests: {len(tests_prod_5)} (via engagements du produit)")
        
        # Findings bruts du product 5
        findings_prod_5_raw = [f for f in findings if f.get('product') == 5]
        print(f"\n   Findings BRUTS du product: {len(findings_prod_5_raw)}")
        
        # Findings enrichis du product 5
        findings_prod_5_enriched = hierarchy.filter_by_product(enriched, 5)
        print(f"   Findings ENRICHIS du product: {len(findings_prod_5_enriched)}")
        
        print(f"\n   Détail par ENGAGEMENT du product 5:")
        for e in engs_prod_5:
            findings_eng = hierarchy.filter_by_engagement(findings_prod_5_enriched, e['id'])
            print(f"      - Engagement {e['id']} ({e['name']}): {len(findings_eng)} findings")
            
            # Afficher les premiers findings
            if findings_eng:
                print(f"         Exemples:")
                for f in findings_eng[:3]:
                    print(f"            • Finding {f['id']:4d} | {f.get('title', 'N/A')[:40]:40s} | test={f.get('test')}")
    
    # ── SECTION 8: RÉSUMÉ & RECOMMANDATIONS ─────────────────────────────────────
    print("\n" + "="*80)
    print("8️⃣  RÉSUMÉ & RECOMMANDATIONS")
    print("="*80)
    
    # Calcul du ratio
    product_5_ratio = len(findings_prod_5_enriched) / len(findings) * 100 if findings else 0
    
    print(f"\n   HIÉRARCHIE COMPLÈTE:")
    print(f"   • {len(products)} produits")
    print(f"   • {len(engagements)} engagements")
    print(f"   • {len(tests)} tests")
    print(f"   • {len(findings)} findings ({findings_with_test} avec test, {findings_without_test} orphelins)")
    
    print(f"\n   PRODUCT 5 SPÉCIFIQUEMENT:")
    print(f"   • {len(engs_prod_5)} engagements")
    print(f"   • {len(tests_prod_5)} tests")
    print(f"   • {len(findings_prod_5_enriched)} findings enrichis ({product_5_ratio:.1f}% du total)")
    
    print(f"\n   ⚠️ SI SEULEMENT 42 FINDINGS S'AFFICHENT:")
    if len(findings_prod_5_enriched) == 42:
        print(f"      ✓ C'EST CORRECT! Tous les {len(findings_prod_5_enriched)} findings du product 5 s'affichent.")
        print(f"      → Il n'y a PAS 458 findings pour product 5, seulement 42!")
    elif len(findings_prod_5_enriched) > 42:
        print(f"      ❌ PROBLÈME: {len(findings_prod_5_enriched)} findings devraient s'afficher, mais seulement 42 le font!")
        print(f"      → C'est un problème du FRONTEND ou du CACHE")
    else:
        print(f"      ✓ Les 42 findings affichés sont corrects")
        print(f"      → Il n'y a que {len(findings_prod_5_enriched)} findings total pour product 5")
    
    print("\n   ÉTAPES DE DÉBOGAGE:")
    print("   1. Vérifier que FindingInput.to_features() inclut toutes les 29 features")
    print("   2. Tester /defectdojo/products/5/findings?limit=500")
    print("   3. Compter les findings retournés par l'API")
    print("   4. Comparer avec /defectdojo/engagements/5/findings")
    print("   5. Vérifier le cache frontend (F12 → Network → Réponse)")
    
    print("\n" + "="*80)
    print("✅ DIAGNOSTIC TERMINÉ")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        diagnose_hierarchy()
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()