"""
DIAGNOSTIC COMPLET: Hiérarchie Product → Engagement → Test → Finding
Affiche la hiérarchie et la distribution des sévérités.
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# Configuration
DEFECTDOJO_URL = os.getenv("DEFECTDOJO_URL", "http://192.168.11.170:8080")
DEFECTDOJO_API_KEY = os.getenv("DEFECTDOJO_API_KEY", "a8506b7874b044ed31f8d6b847ca4d6b15bdb868")
HEADERS = {"Authorization": f"Token {DEFECTDOJO_API_KEY}"}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def get_json(endpoint: str, params: dict = None):
    """Récupère les données paginées de l'API DefectDojo."""
    url = f"{DEFECTDOJO_URL}/api/v2/{endpoint}"
    results = []
    while url:
        resp = SESSION.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        results.extend(data.get("results", []))
        url = data.get("next")
        params = None  # next contient déjà les paramètres
    return results


def diagnose_hierarchy():
    print("\n" + "=" * 80)
    print("🔍 DIAGNOSTIC: HIÉRARCHIE PRODUCT → ENGAGEMENT → TEST → FINDING")
    print("=" * 80)

    print("\n📥 Récupération des données depuis DefectDojo...")
    products = get_json("products")
    engagements = get_json("engagements")
    tests = get_json("tests")
    findings = get_json("findings", params={"limit": 2000, "include_inactive": True})

    print(f"   ✓ {len(products)} produits")
    print(f"   ✓ {len(engagements)} engagements")
    print(f"   ✓ {len(tests)} tests")
    print(f"   ✓ {len(findings)} findings")

    # ── Distribution des sévérités (brutes, tous produits) ─────────────────────
    print("\n" + "=" * 80)
    print("📊 DISTRIBUTION DES SÉVÉRITÉS (brutes, tous produits)")
    print("=" * 80)
    severity_counts = defaultdict(int)
    for f in findings:
        sev = f.get("severity", "Unknown")
        severity_counts[sev] += 1
    for sev, cnt in sorted(severity_counts.items(), key=lambda x: -x[1]):
        print(f"   {sev.upper():12s} : {cnt:4d}")

    # ── SECTION 1: PRODUITS ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("1️⃣  PRODUITS")
    print("=" * 80)
    for p in products:
        print(f"\n   ID={p['id']:2d} | {p.get('name', 'N/A'):30s}")

    # ── SECTION 2: ENGAGEMENTS ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("2️⃣  ENGAGEMENTS")
    print("=" * 80)
    print("\n   ID | Nom                            | Produit")
    print("   " + "-" * 70)
    for e in engagements:
        prod_name = next((p.get("name", "N/A") for p in products if p["id"] == e.get("product")), "?")
        print(f"   {e['id']:2d} | {e.get('name', 'N/A'):30s} | {e.get('product'):2d} ({prod_name})")

    # ── SECTION 3: TESTS → ENGAGEMENTS ────────────────────────────────────────
    print("\n" + "=" * 80)
    print("3️⃣  TESTS → ENGAGEMENTS")
    print("=" * 80)
    tests_by_engagement = defaultdict(list)
    tests_orphaned = 0
    for t in tests:
        eng_id = t.get("engagement")
        if eng_id is None:
            tests_orphaned += 1
        else:
            tests_by_engagement[eng_id].append(t["id"])
    print(f"\n   Total: {len(tests)} tests")
    print(f"   Orphelins (sans engagement): {tests_orphaned}")
    print("\n   Engagement | Nom           | Produit | # Tests")
    print("   " + "-" * 70)
    for eng_id in sorted(tests_by_engagement.keys()):
        eng = next((e for e in engagements if e["id"] == eng_id), {})
        prod_id = eng.get("product", "?")
        prod_name = next((p.get("name", "?") for p in products if p["id"] == prod_id), "?")
        num_tests = len(tests_by_engagement[eng_id])
        print(f"   {eng_id:10d} | {eng.get('name', 'N/A'):13s} | {prod_id:7} | {num_tests:8d}")

    # ── SECTION 4: FINDINGS BRUTS ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("4️⃣  FINDINGS BRUTS")
    print("=" * 80)
    findings_with_test = sum(1 for f in findings if f.get("test") is not None)
    findings_without_test = len(findings) - findings_with_test
    findings_by_engagement = defaultdict(list)
    findings_by_product = defaultdict(list)
    for f in findings:
        eng_id = f.get("engagement")
        prod_id = f.get("product")
        if eng_id:
            findings_by_engagement[eng_id].append(f["id"])
        if prod_id:
            findings_by_product[prod_id].append(f["id"])

    print(f"\n   Total: {len(findings)} findings")
    print(f"   Avec test_id valide: {findings_with_test}")
    print(f"   SANS test_id (orphelins): {findings_without_test} ⚠️")

    print("\n   Par PRODUCT:")
    print("   Produit | Nom                    | # Findings")
    print("   " + "-" * 70)
    for prod_id in sorted(findings_by_product.keys()):
        prod = next((p for p in products if p["id"] == prod_id), {})
        num_findings = len(findings_by_product[prod_id])
        print(f"   {prod_id:7} | {prod.get('name', 'N/A'):22s} | {num_findings:11d}")

    print("\n   Par ENGAGEMENT:")
    print("   Engagement | Nom           | Product | # Findings")
    print("   " + "-" * 70)
    for eng_id in sorted(findings_by_engagement.keys()):
        eng = next((e for e in engagements if e["id"] == eng_id), {})
        prod_id = eng.get("product", "?")
        num_findings = len(findings_by_engagement[eng_id])
        print(f"   {eng_id:10} | {eng.get('name', 'N/A'):13s} | {prod_id:7} | {num_findings:11d}")

    # ── SECTION 5: FOCUS PRODUCT 5 (InvisiThreat) ──────────────────────────────
    print("\n" + "=" * 80)
    print("7️⃣  FOCUS: PRODUCT 5 (InvisiThreat)")
    print("=" * 80)
    product_5 = next((p for p in products if p["id"] == 5), None)
    if not product_5:
        print("   ❌ Product 5 non trouvé!")
    else:
        print(f"\n   Produit: {product_5['name']}")
        engs_prod_5 = [e for e in engagements if e.get("product") == 5]
        print(f"\n   Engagements: {len(engs_prod_5)}")
        for e in engs_prod_5:
            print(f"      - ID={e['id']:2d}: {e['name']}")

        findings_prod_5_raw = [f for f in findings if f.get("product") == 5]
        print(f"\n   Findings BRUTS du product: {len(findings_prod_5_raw)}")

        # Distribution des sévérités pour le product 5
        print("\n   DISTRIBUTION DES SÉVÉRITÉS (Product 5, bruts) :")
        sev_prod5 = defaultdict(int)
        for f in findings_prod_5_raw:
            sev = f.get("severity", "Unknown")
            sev_prod5[sev] += 1
        for sev, cnt in sorted(sev_prod5.items(), key=lambda x: -x[1]):
            print(f"      {sev.upper():12s} : {cnt:4d}")

    # ── SECTION 6: RÉSUMÉ ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("8️⃣  RÉSUMÉ")
    print("=" * 80)
    product_5_total = len([f for f in findings if f.get("product") == 5])
    print(f"\n   • Total findings (tous produits) : {len(findings)}")
    print(f"   • Findings pour le produit 5 : {product_5_total}")
    print("\n" + "=" * 80)
    print("✅ DIAGNOSTIC TERMINÉ")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        diagnose_hierarchy()
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()