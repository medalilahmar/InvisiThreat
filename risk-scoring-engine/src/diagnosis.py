import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def check_raw_data():
    print("\n" + "="*70)
    print("DIAGNOSTIC 1: DONNÉES BRUTES")
    print("="*70)
    
    findings = pd.read_csv(RAW_DIR / "findings_raw.csv")
    
    if "severity" in findings.columns:
        print("\n✓ Colonne 'severity' trouvée")
        print(findings["severity"].value_counts())
        print(f"\nUnique values: {findings['severity'].unique()}")
    else:
        print("Colonne 'severity' NOT FOUND")
    
    if "severity" in findings.columns:
        high_critical = findings[findings["severity"].isin(["High", "Critical"])]
        print(f"\nFindings High/Critical: {len(high_critical)} / {len(findings)}")
        print(f"Percentage: {len(high_critical)/len(findings)*100:.1f}%")
    
    return findings

def check_processed_data():
    print("\n" + "="*70)
    print("DIAGNOSTIC 2: DONNÉES PRÉPROCESSÉES")
    print("="*70)
    
    df = pd.read_csv(PROCESSED_DIR / "findings_clean.csv")
    
    print(f"\nTotal rows: {len(df)}")
    print(f"Total cols: {len(df.columns)}")
    
    if "risk_class" in df.columns:
        print("\n✓ risk_class distribution:")
        print(df["risk_class"].value_counts().sort_index())
        valid = df["risk_class"].notna().sum()
        print(f"\nValid samples: {valid} / {len(df)}")
        print(f"Excluded samples: {len(df) - valid}")
    
    if "severity_num" in df.columns:
        print("\n✓ severity_num distribution:")
        print(df["severity_num"].value_counts().sort_index())
    
    print(f"\n✓ Features présentes: {len([c for c in df.columns if not c.startswith('_')])}")
    
    if "severity" in df.columns:
        high_critical = df[df["severity"].isin(["High", "Critical"])]
        print(f"\n  High/Critical findings: {len(high_critical)}")
        if len(high_critical) > 0:
            print("  Risk class distribution for High/Critical:")
            print(high_critical["risk_class"].value_counts())
    
    return df

def check_feature_separation():
    print("\n" + "="*70)
    print("DIAGNOSTIC 3: SÉPARATION DES FEATURES")
    print("="*70)
    
    df = pd.read_csv(PROCESSED_DIR / "findings_clean.csv")
    
    key_features = ["cvss_score", "epss_score", "age_days", "exploit_risk"]
    
    for feature in key_features:
        if feature in df.columns:
            print(f"\n✓ {feature}:")
            valid = df[df["risk_class"].notna()]
            for cls in sorted(valid["risk_class"].unique()):
                subset = valid[valid["risk_class"] == cls][feature]
                print(f"  Class {int(cls)}: mean={subset.mean():.4f}, std={subset.std():.4f}, min={subset.min():.4f}, max={subset.max():.4f}")

def check_label_quality():
    print("\n" + "="*70)
    print("DIAGNOSTIC 4: QUALITÉ DU LABEL")
    print("="*70)
    
    df = pd.read_csv(PROCESSED_DIR / "findings_clean.csv")
    
    if "days_to_fix" in df.columns:
        print(f"\ndays_to_fix coverage: {df['days_to_fix'].notna().sum()} / {len(df)}")
        print(f"Percentage: {df['days_to_fix'].notna().mean()*100:.1f}%")
        
        if "label_source" in df.columns:
            print("\nLabel sources distribution:")
            print(df["label_source"].value_counts())
    
    if "score_composite_raw" in df.columns and "risk_class" in df.columns:
        print("\nComposite score vs risk_class:")
        valid = df[df["risk_class"].notna()]
        for cls in sorted(valid["risk_class"].unique()):
            subset = valid[valid["risk_class"] == cls]["score_composite_raw"]
            print(f"  Class {int(cls)}: mean={subset.mean():.4f}, std={subset.std():.4f}")

def check_data_leakage():
    print("\n" + "="*70)
    print("DIAGNOSTIC 5: DATA LEAKAGE CHECK")
    print("="*70)
    
    df = pd.read_csv(PROCESSED_DIR / "findings_clean.csv")
    
    FEATURE_COLS = [
        "cvss_score", "cvss_score_norm", "age_days", "age_days_norm",
        "has_cve", "has_cwe", "tags_count", "tags_count_norm",
        "tag_urgent", "tag_in_production", "tag_sensitive", "tag_external",
        "product_fp_rate", "cvss_x_has_cve", "age_x_cvss",
        "epss_score", "epss_percentile", "has_high_epss", "epss_x_cvss", "epss_score_norm",
        "exploit_risk", "context_score", "days_open_high",
    ]
    
    EXCLUDE_COLS = [
        "days_to_fix", "risk_class", "risk_score",
        "is_mitigated", "out_of_scope", "is_false_positive",
        "label_source", "score_composite_raw", "score_composite_adj"
    ]
    
    leakage = [c for c in EXCLUDE_COLS if c in FEATURE_COLS]
    if leakage:
        print(f" LEAKAGE DETECTED: {leakage}")
    else:
        print(" NO LEAKAGE DETECTED")
    
    print(f"\nFeature columns: {len(FEATURE_COLS)}")
    print(f"Excluded columns: {len(EXCLUDE_COLS)}")
    print(f"Features present in CSV: {len([c for c in FEATURE_COLS if c in df.columns])}")

def suggest_improvements():
    print("\n" + "="*70)
    print("SUGGESTIONS D'AMÉLIORATION")
    print("="*70)
    
    df = pd.read_csv(PROCESSED_DIR / "findings_clean.csv")
    findings = pd.read_csv(RAW_DIR / "findings_raw.csv")
    
    print("\n1️  AUGMENTER LA COMPLEXITÉ")
    print("   • Ajouter plus de données High/Critical")
    print("   • Utiliser k-fold stratifié (actuellement 25% test)")
    print("   • Réduire test size à 15-20%")
    
    print("\n2️  VALIDER LA GÉNÉRALISATION")
    print("   • Créer un ensemble de validation indépendant")
    print("   • Tester sur données futures")
    print("   • Faire cross-validation complète")
    
    print("\n3️  VÉRIFIER LES DONNÉES BRUTES")
    high_critical_raw = findings[findings["severity"].isin(["High", "Critical"])]
    print(f"   • High/Critical dans raw: {len(high_critical_raw)}")
    print(f"   • Ratio High/Critical: {len(high_critical_raw)/len(findings)*100:.1f}%")
    
    print("\n4️  AMÉLIORATIONS POSSIBLES")
    print("   • Ajouter class_weight imbalancé")
    print("   • Utiliser SMOTE pour sur-échantillonner les classes minoritaires")
    print("   • Augmenter la profondeur de tuning")

if __name__ == "__main__":
    try:
        findings = check_raw_data()
        df = check_processed_data()
        check_feature_separation()
        check_label_quality()
        check_data_leakage()
        suggest_improvements()
        
        print("\n" + "="*70)
        print(" DIAGNOSTIC COMPLET TERMINÉ")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f" Erreur: {e}")
        import traceback
        traceback.print_exc()