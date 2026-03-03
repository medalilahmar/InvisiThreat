import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import joblib

import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

class TestTrain:
    """Tests pour train.py"""
    
    def test_feature_selection(self, sample_processed_dataframe):
        """Test que les bonnes features sont sélectionnées"""
        df = sample_processed_dataframe
        
        # Liste des features attendues (à adapter selon votre train.py)
        expected_features = [
            'severity_num', 'cvss_score', 'age_days', 'has_cve', 'has_cwe',
            'tags_count', 'is_false_positive', 'is_active',
            'tag_urgent', 'tag_in_production', 'tag_sensitive', 'tag_external',
            'severity_x_active', 'product_fp_rate', 'cvss_severity_gap',
            'cvss_x_severity', 'cvss_x_has_cve', 'severity_x_urgent', 'age_x_cvss',
            'cvss_score_norm', 'severity_norm', 'age_days_norm',
            'tags_count_norm', 'cvss_severity_gap_norm'
        ]
        
        for feat in expected_features:
            assert feat in df.columns, f"Feature manquante: {feat}"
    
    def test_target_distribution(self, sample_processed_dataframe):
        """Test la distribution des classes cibles"""
        df = sample_processed_dataframe
        y = df['risk_class']
        
        unique_classes = sorted(y.unique())
        assert set(unique_classes).issubset({0, 1, 2, 3, 4})
        
        class_counts = y.value_counts()
        min_count = class_counts.min()
        max_count = class_counts.max()
        
        assert max_count / min_count < 10
    
    def test_train_test_split(self, sample_processed_dataframe):
        """Test que le split train/test préserve la distribution"""
        df = sample_processed_dataframe
        y = df['risk_class']
        
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            df, y, test_size=0.2, random_state=42, stratify=y
        )
        
        assert len(X_train) == int(len(df) * 0.8)
        assert len(X_test) == int(len(df) * 0.2)
        
        train_dist = y_train.value_counts(normalize=True)
        test_dist = y_test.value_counts(normalize=True)
        
        for cls in range(5):
            if cls in train_dist and cls in test_dist:
                assert abs(train_dist[cls] - test_dist[cls]) < 0.1
    
    def test_model_training(self, sample_processed_dataframe):
        """Test que l'entraînement produit un modèle"""
        from sklearn.ensemble import RandomForestClassifier
        
        df = sample_processed_dataframe
        
        feature_cols = [c for c in df.columns if c != 'risk_class']
        X = df[feature_cols]
        y = df['risk_class']
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        assert hasattr(model, 'estimators_')
        assert len(model.estimators_) == 10
        
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)
    
    def test_model_persistence(self, sample_processed_dataframe, tmp_path):
        """Test la sauvegarde et chargement du modèle"""
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        
        df = sample_processed_dataframe
        feature_cols = [c for c in df.columns if c != 'risk_class']
        X = df[feature_cols]
        y = df['risk_class']
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        model_path = tmp_path / "test_model.pkl"
        joblib.dump(model, model_path)
        
        loaded_model = joblib.load(model_path)
        
        y_pred_orig = model.predict(X)
        y_pred_loaded = loaded_model.predict(X)
        
        assert np.array_equal(y_pred_orig, y_pred_loaded)
    
    def test_metrics_calculation(self, sample_processed_dataframe):
        """Test le calcul des métriques"""
        from sklearn.metrics import f1_score, accuracy_score
        
        df = sample_processed_dataframe
        feature_cols = [c for c in df.columns if c != 'risk_class']
        X = df[feature_cols]
        y_true = df['risk_class']
        
        y_pred = y_true.copy()
        
        accuracy = accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        assert accuracy == 1.0
        assert f1_weighted == 1.0
        assert f1_macro == 1.0
        
        y_pred_bad = pd.Series([0] * len(y_true))
        f1_weighted_bad = f1_score(y_true, y_pred_bad, average='weighted')
        
        assert f1_weighted_bad < 1.0
    
    def test_cross_validation(self, sample_processed_dataframe):
        """Test la validation croisée"""
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        
        df = sample_processed_dataframe
        feature_cols = [c for c in df.columns if c != 'risk_class']
        X = df[feature_cols]
        y = df['risk_class']
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
        
        assert len(scores) == 5
        assert all(0 <= s <= 1 for s in scores)
        assert scores.mean() > 0
    
    def test_feature_importance(self, sample_processed_dataframe):
        """Test que les feature importances sont calculées"""
        from sklearn.ensemble import RandomForestClassifier
        
        df = sample_processed_dataframe
        feature_cols = [c for c in df.columns if c != 'risk_class']
        X = df[feature_cols]
        y = df['risk_class']
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        importances = model.feature_importances_
        
        assert len(importances) == len(feature_cols)
        assert all(0 <= imp <= 1 for imp in importances)
        assert abs(sum(importances) - 1.0) < 0.01  # Somme ≈ 1
    
    def test_metadata_generation(self, sample_processed_dataframe, tmp_path):
        """Test la génération des métadonnées"""
        from sklearn.ensemble import RandomForestClassifier
        import json
        from datetime import datetime
        
        df = sample_processed_dataframe
        feature_cols = [c for c in df.columns if c != 'risk_class']
        X = df[feature_cols]
        y = df['risk_class']
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'n_classes': len(model.classes_),
            'classes': model.classes_.tolist(),
            'n_features': len(feature_cols),
            'features': feature_cols,
            'metrics': {
                'f1_weighted': 0.95,
                'f1_macro': 0.94
            }
        }
        
        meta_path = tmp_path / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)
        
        with open(meta_path) as f:
            loaded_meta = json.load(f)
        
        assert loaded_meta['n_classes'] == metadata['n_classes']
        assert loaded_meta['n_features'] == metadata['n_features']
        assert len(loaded_meta['features']) == len(feature_cols)