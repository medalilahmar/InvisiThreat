"""
Diagnostic: Trouver les 214 findings manquants pour Product 5
"""

import os
import sys
from pathlib import Path
from collections import defaultdict


from dotenv import load_dotenv
load_dotenv()

from api import DefectDojoClient  

client = DefectDojoClient(
    os.getenv("DEFECTDOJO_URL", "http://192.168.11.170:8080"),
    os.getenv("DEFECTDOJO_API_KEY", "a8506b7874b044ed31f8d6b847ca4d6b15bdb868"),
)

print("=" * 80)
print("🔍 DIAGNOSTIC: OÙ SONT LES 214 FINDINGS MANQUANTS?")
print("=" * 80)

print("\n📥 Récupération des données...")
products = client.get_products()
engagements = client.get_engagements()
tests = client.get_tests()
findings = client.get_findings(limit=2000) 

print(f"✓ {len(products)} produits")
print(f"✓ {len(engagements)} engagements")
print(f"✓ {len(tests)} tests")
print(f"✓ {len(findings)} findings TOTAUX")


print("\n" + "=" * 80)
print("🎯 PRODUCT 5 (InvisiThreat)")
print("=" * 80)

product_5 = next((p for p in products if p['id'] == 5), None)
if not product_5:
    print("❌ Product 5 non trouvé!")
    sys.exit(1)

print(f"\nProduit: {product_5['name']}")

engs_5 = [e for e in engagements if e.get('product') == 5]
print(f"\nEngagements: {len(engs_5)}")
for e in engs_5:
    print(f"  - ID={e['id']}: {e['name']}")

eng_ids_5 = [e['id'] for e in engs_5]
tests_5 = [t for t in tests if t.get('engagement') in eng_ids_5]
print(f"\nTests: {len(tests_5)}")
for t in tests_5[:10]:
    print(f"  - ID={t['id']}: {t.get('name', 'N/A')}")

print("\n" + "-" * 80)
print("FINDINGS BRUTS (via engagement_id dans DefectDojo)")
print("-" * 80)

findings_by_product_raw = defaultdict(list)
findings_by_engagement_raw = defaultdict(list)
findings_by_severity = defaultdict(int)
findings_by_status = defaultdict(int)

for f in findings:
    prod_id = f.get('product')
    eng_id = f.get('engagement')
    severity = f.get('severity', 'unknown')
    
    if prod_id == 5:
        findings_by_product_raw[prod_id].append(f['id'])
    
    if eng_id in eng_ids_5:
        findings_by_engagement_raw[eng_id].append(f['id'])
    
    if prod_id == 5:
        findings_by_severity[severity] += 1
        
        is_active = f.get('is_active', False)
        is_mitigated = f.get('is_mitigated', False)
        false_positive = f.get('false_positive', False)
        out_of_scope = f.get('out_of_scope', False)
        
        if is_mitigated:
            findings_by_status['mitigated'] += 1
        elif false_positive:
            findings_by_status['false_positive'] += 1
        elif out_of_scope:
            findings_by_status['out_of_scope'] += 1
        elif is_active:
            findings_by_status['active'] += 1
        else:
            findings_by_status['unknown'] += 1

print(f"\nFindings du Product 5 (field 'product'=5): {len(findings_by_product_raw[5])}")

print(f"\nFindings du Engagement 5 (field 'engagement'=5): {len(findings_by_engagement_raw[5])}")
for eng_id, fids in findings_by_engagement_raw.items():
    if eng_id in eng_ids_5:
        eng = next((e for e in engs_5 if e['id'] == eng_id), {})
        print(f"  - Engagement {eng_id} ({eng.get('name')}): {len(fids)} findings")

print(f"\n✓ Distribution par SEVERITY (Product 5):")
for severity, count in sorted(findings_by_severity.items(), key=lambda x: -x[1]):
    print(f"  - {severity.upper():15s}: {count:3d}")

print(f"\n✓ Distribution par STATUS (Product 5):")
for status, count in sorted(findings_by_status.items(), key=lambda x: -x[1]):
    print(f"  - {status.upper():15s}: {count:3d}")


print("\n" + "=" * 80)
print("🔄 COMPARER AVEC API")
print("=" * 80)

api_findings = client.get_findings(engagement_id=5, limit=1000)
print(f"\nAPI GET /findings (engagement_id=5): {len(api_findings)} findings")

api_by_severity = defaultdict(int)
api_by_status = defaultdict(int)

for f in api_findings:
    severity = f.get('severity', 'unknown')
    api_by_severity[severity] += 1
    
    is_active = f.get('is_active', False)
    is_mitigated = f.get('is_mitigated', False)
    false_positive = f.get('false_positive', False)
    
    if is_mitigated:
        api_by_status['mitigated'] += 1
    elif false_positive:
        api_by_status['false_positive'] += 1
    elif is_active:
        api_by_status['active'] += 1
    else:
        api_by_status['unknown'] += 1

print(f"\n✓ Distribution par SEVERITY (API):")
for severity, count in sorted(api_by_severity.items(), key=lambda x: -x[1]):
    print(f"  - {severity.upper():15s}: {count:3d}")

print(f"\n✓ Distribution par STATUS (API):")
for status, count in sorted(api_by_status.items(), key=lambda x: -x[1]):
    print(f"  - {status.upper():15s}: {count:3d}")


print("\n" + "=" * 80)
print("📊 RÉSUMÉ")
print("=" * 80)

print(f"\nDefectDojo UI affiche: 256 findings")
print(f"API retourne: {len(api_findings)} findings")
print(f"Manquants: {256 - len(api_findings)} findings")

if len(api_findings) < 256:
    print(f"\n❌ PROBLÈME: {256 - len(api_findings)} findings manquants!")
    print(f"\nCauses possibles:")
    print(f"  1. Findings avec is_active=false (inactifs)")
    print(f"  2. Findings is_mitigated=true (résolus)")
    print(f"  3. Findings false_positive=true")
    print(f"  4. Findings out_of_scope=true")
    print(f"  5. Findings sans engagement_id (orphelins)")
else:
    print(f"\n✅ CORRECT! Tous les findings sont retournés.")

print("\n" + "=" * 80 + "\n")