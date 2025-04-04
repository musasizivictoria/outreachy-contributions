"""
External validation script for permeability models.
Tests model performance on unseen datasets from various sources.
"""

import os
import json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import requests
from typing import Dict, List, Tuple

class ExternalValidator:
    def __init__(self, model_path: str, threshold: float = 8e-6):
        """Initialize validator with model and threshold."""
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self.threshold = threshold
        
    def generate_morgan_fingerprints(self, smiles: str) -> np.ndarray:
        """Generate Morgan fingerprints for a SMILES string."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
        except:
            return None

    def predict_batch(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict permeability class and probability for a batch of compounds."""
        probs = self.model.predict_proba(features)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return preds, probs

    def validate_dataset(self, data: pd.DataFrame, features: np.ndarray, smiles_col: str, label_col: str) -> Dict:
        """Validate model on a dataset."""
        # Get predictions
        y_pred, y_prob = self.predict_batch(features)
        y_true = data[label_col].values
        
        # Store results
        compounds = []
        for i, row in data.iterrows():
            compounds.append({
                'SMILES': row[smiles_col],
                'Name': row.get('Name', 'Unknown'),
                'ChEMBL_ID': row.get('ChEMBL_ID', 'Unknown'),
                'True_Label': y_true[i],
                'Predicted_Label': y_pred[i],
                'Probability': y_prob[i]
            })
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'n_compounds': len(y_true),
            'compounds': compounds
        }
        
        return metrics

def load_ich_reference_data() -> pd.DataFrame:
    """Load ICH reference compounds dataset."""
    # Example format - replace with actual data source
    data = {
        'SMILES': [
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CC1=CC=C(C=C1)NC(=O)CCCN',  # Procainamide
            'CC1=C(C=C(C=C1)O)C(CCNC(C)C)O'  # Metoprolol
        ],
        'Name': ['Aspirin', 'Procainamide', 'Metoprolol'],
        'Permeability': [1, 0, 1]  # High/Low
    }
    return pd.DataFrame(data)

def load_chembl_caco2_data() -> pd.DataFrame:
    """Load Caco-2 data from ChEMBL."""
    try:
        df = pd.read_csv('data/external_validation/chembl_caco2.csv')
        print(f"Loaded {len(df)} compounds from ChEMBL")
        return df
    except Exception as e:
        print(f"Error loading ChEMBL data: {e}")
        return pd.DataFrame()

def main():
    # Initialize validator
    validator = ExternalValidator('models/morgan/xgboost_model.json')
    
    # Load improved validation data
    balanced_data = pd.read_csv('data/processed/caco2_test_improved.csv')
    print(f"\nValidation set: Improved ChEMBL dataset")
    print(f"Size: {len(balanced_data)} compounds")
    print("\nClass distribution:")
    print(f"High permeability: {(balanced_data['Permeability'] == 1).sum()} ({(balanced_data['Permeability'] == 1).mean():.1%})")
    print(f"Low permeability: {(balanced_data['Permeability'] == 0).sum()} ({(balanced_data['Permeability'] == 0).mean():.1%})")
    print("\nNote: Zero permeability values removed, threshold adjusted, and classes balanced")
    print(f"\nGenerating Morgan fingerprints for {len(balanced_data)} balanced validation compounds...")
    
    # Generate fingerprints
    features = []
    valid_indices = []
    for idx, row in balanced_data.iterrows():
        fp = validator.generate_morgan_fingerprints(row['SMILES'])
        if fp is not None:
            features.append(fp)
            valid_indices.append(idx)
    
    features = np.array(features)
    balanced_data_valid = balanced_data.iloc[valid_indices].reset_index(drop=True)
    
    print(f"Generated fingerprints for {len(features)} compounds")
    
    # Validate on balanced data
    print("\nValidating model...")
    balanced_metrics = validator.validate_dataset(
        balanced_data,
        features,
        smiles_col='SMILES',
        label_col='Permeability'
    )
    
    # Print metrics
    print(f"\nValidation Results:")
    print(f"Number of compounds: {balanced_metrics['n_compounds']}")
    print(f"Accuracy: {balanced_metrics['accuracy']:.3f}")
    print(f"ROC-AUC: {balanced_metrics['roc_auc']:.3f}")
    print(f"Precision: {balanced_metrics['precision']:.3f}")
    print(f"Recall: {balanced_metrics['recall']:.3f}")
    print(f"F1-score: {balanced_metrics['f1']:.3f}")
    
    # Analyze predictions by permeability range
    compounds = pd.DataFrame(balanced_metrics['compounds'])
    compounds['Original_Value'] = balanced_data['Permeability_Value']
    compounds['Original_Units'] = balanced_data['Original_Units']
    
    # Convert values to cm/s for consistent ranges
    def convert_to_cm_s(row):
        value = row['Original_Value']
        units = row['Original_Units']
        if units == 'nm/s':
            return value * 1e-7
        elif units == '10^-6 cm/s':
            return value * 1e-6
        return value
    
    compounds['Permeability_cm_s'] = compounds.apply(convert_to_cm_s, axis=1)
    
    # Group predictions by permeability ranges
    def get_perm_range(x):
        if x < 1e-7: return '< 0.1 µm/s'
        elif x < 1e-6: return '0.1-1 µm/s'
        elif x < 1e-5: return '1-10 µm/s'
        elif x < 1e-4: return '10-100 µm/s'
        else: return '> 100 µm/s'
    
    compounds['Perm_Range'] = compounds['Permeability_cm_s'].apply(get_perm_range)
    
    print('\nPrediction Analysis by Permeability Range:')
    range_metrics = compounds.groupby('Perm_Range').agg({
        'True_Label': 'count',
        'Predicted_Label': lambda x: (x == compounds.loc[x.index, 'True_Label']).mean()
    }).round(3)
    range_metrics.columns = ['Count', 'Accuracy']
    print(range_metrics)
    
    # Save detailed results
    os.makedirs('data/external_validation', exist_ok=True)
    
    # Save metrics
    results = {
        'balanced': {k: v for k, v in balanced_metrics.items() if k != 'compounds'},
        'timestamp': pd.Timestamp.now().isoformat()
    }
    with open('data/external_validation/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    compounds.to_csv('data/external_validation/predictions.csv', index=False)
    
    # Save features
    np.save('data/external_validation/chembl_features.npy', features)
    
    print("\nResults saved to:")
    print("- data/external_validation/results.json")
    print("- data/external_validation/predictions.csv")
    print("- data/external_validation/chembl_features.npy")

if __name__ == "__main__":
    main()
