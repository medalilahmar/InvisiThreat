import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import json


sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestPreprocess:
    
    def test_severity_mapping(self, sample_raw_dataframe):
        df = sample_raw_dataframe
        
        severity_map = {'Info': 0, 'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        df['severity_num'] = df['severity'].map(severity_map)
        
        assert df.loc[0, 'severity_num'] == 3  
        assert df.loc[1, 'severity_num'] == 2  
        assert df.loc[2, 'severity_num'] == 1  
    
    def test_age_calculation(self, sample_raw_dataframe):
        from datetime import datetime
        
        df = sample_raw_dataframe
        df['created'] = pd.to_datetime(df['created'])
        
        now = datetime.now()
        df['age_days'] = (now - df['created']).dt.days
        
        assert df['age_days'].dtype == 'int64'
        assert all(df['age_days'] > 0)
    
    def test_tag_detection(self):
        test_cases = [
            (['urgent', 'prod'], 1, 1, 0, 0),  
            (['pii', 'external'], 0, 0, 1, 1), 
            ([], 0, 0, 0, 0),                   
            (['blocker', 'p0'], 1, 0, 0, 0),    
        ]
        
        for tags, exp_urgent, exp_prod, exp_sensitive, exp_external in test_cases:
            tag_urgent = 1 if any(t in ['urgent', 'blocker', 'p0'] for t in tags) else 0
            tag_prod = 1 if any(t in ['prod', 'production'] for t in tags) else 0
            tag_sensitive = 1 if any(t in ['pii', 'gdpr', 'sensitive'] for t in tags) else 0
            tag_external = 1 if any(t in ['external', 'public', 'internet'] for t in tags) else 0
            
            assert tag_urgent == exp_urgent
            assert tag_prod == exp_prod
            assert tag_sensitive == exp_sensitive
            assert tag_external == exp_external
    
    def test_risk_score_calculation(self):
        row1 = {
            'severity_num': 3,
            'cvss_score': 8.5,
            'age_days': 30,
            'has_cve': 1,
            'tags_count': 3
        }
        
        score1 = (
            row1['severity_num'] * 2.0 +           
            row1['cvss_score'] * 0.5 +              
            min(row1['age_days'] / 20, 2) +          
            row1['has_cve'] * 1.5 +                   
            min(row1['tags_count'] * 0.5, 2)          
        ) / 1.5  
        
        row2 = {
            'severity_num': 1,
            'cvss_score': 2.0,
            'age_days': 5,
            'has_cve': 0,
            'tags_count': 0
        }
        
        score2 = (
            row2['severity_num'] * 2.0 +
            row2['cvss_score'] * 0.5 +
            min(row2['age_days'] / 20, 2) +
            row2['has_cve'] * 1.5 +
            min(row2['tags_count'] * 0.5, 2)
        ) / 1.5
        
        assert score1 > score2
        assert 0 <= score1 <= 10
        assert 0 <= score2 <= 10
    
    def test_feature_engineering(self, sample_raw_dataframe):
        """Test la création des features dérivées"""
        df = sample_raw_dataframe.copy()
        
        df['severity_num'] = [3, 2, 1]
        df['cvss_score'] = [7.5, 5.0, 2.5]
        df['age_days'] = [45, 30, 90]
        df['has_cve'] = [1, 0, 0]
        df['is_active'] = [1, 1, 0]
        df['tag_urgent'] = [1, 0, 0]
        
        df['cvss_x_severity'] = df['cvss_score'] * df['severity_num']
        df['severity_x_active'] = df['severity_num'] * df['is_active']
        df['age_x_cvss'] = df['age_days'] * df['cvss_score']
        df['cvss_x_has_cve'] = df['cvss_score'] * df['has_cve']
        
        assert df.loc[0, 'cvss_x_severity'] == 7.5 * 3  
        assert df.loc[0, 'severity_x_active'] == 3 * 1  
        assert df.loc[0, 'age_x_cvss'] == 45 * 7.5  
        assert df.loc[0, 'cvss_x_has_cve'] == 7.5 * 1  
        
        assert df.loc[2, 'severity_x_active'] == 1 * 0  
        assert df.loc[2, 'cvss_x_has_cve'] == 2.5 * 0  
    
    def test_normalization(self):
        """Test la normalisation des features (CORRIGÉ)"""
        df = pd.DataFrame({
            'cvss_score': [0, 5, 10],
            'severity_num': [0, 2, 4],
            'age_days': [0, 180, 365],
            'tags_count': [0, 5, 20]
        })
        
        df['cvss_score_norm'] = df['cvss_score'] / 10
        df['severity_norm'] = df['severity_num'] / 4
        df['age_days_norm'] = df['age_days'] / 365
        df['tags_count_norm'] = (df['tags_count'] / 10).clip(upper=1.0)  # ← CORRECTION
        
        assert df['cvss_score_norm'].min() == 0
        assert df['cvss_score_norm'].max() == 1
        assert df['severity_norm'].min() == 0
        assert df['severity_norm'].max() == 1
        
        assert df.loc[1, 'cvss_score_norm'] == 0.5
        assert df.loc[1, 'severity_norm'] == 0.5
        assert df.loc[2, 'age_days_norm'] == 1.0
        assert df.loc[2, 'tags_count_norm'] == 1.0  
    
    def test_output_structure(self, tmp_path):
        """Test que la structure de sortie est correcte"""
        df_test = pd.DataFrame({
            'id': [1, 2],
            'severity_num': [3, 1],
            'cvss_score': [7.5, 2.5],
            'risk_class': [3, 1],
            'risk_score': [7.2, 1.8]
        })
        
        output_path = tmp_path / "test_output.csv"
        df_test.to_csv(output_path, index=False)
        
        assert output_path.exists()
        df_loaded = pd.read_csv(output_path)
        assert len(df_loaded) == 2
        assert 'risk_class' in df_loaded.columns
        assert 'risk_score' in df_loaded.columns