"""
Retrain the model with balanced data using a better permeability threshold.
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_process_data(file_path, threshold=8e-6):
    """Load and process data with permeability threshold."""
    df = pd.read_csv(file_path)
    
    # Convert permeability values if needed
    if 'Y' in df.columns:
        df['Permeability_cm_s'] = 10 ** df['Y']
    else:
        def convert_to_cm_s(row):
            value = row['Permeability_Value']
            units = row['Original_Units']
            if units == 'nm/s':
                return value * 1e-7
            elif units == '10^-6 cm/s':
                return value * 1e-6
            elif units == 'um/s':
                return value * 1e-4
            return value
        df['Permeability_cm_s'] = df.apply(convert_to_cm_s, axis=1)
    
    # Remove invalid values
    df = df[df['Permeability_cm_s'] > 0].copy()
    
    # Calculate log permeability
    df['Log_Perm'] = np.log10(df['Permeability_cm_s'])
    
    # Apply threshold
    df['Permeability'] = (df['Permeability_cm_s'] >= threshold).astype(int)
    
    return df

def generate_morgan_fingerprints(smiles_list, radius=2, nBits=2048):
    """Generate Morgan fingerprints for a list of SMILES."""
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
            fingerprints.append(list(fp.ToBitString()))
        else:
            fingerprints.append([0] * nBits)
    return np.array(fingerprints, dtype=int)

def create_balanced_dataset(df, random_state=42):
    """Create a balanced dataset by undersampling the majority class."""
    min_class_size = min(
        (df['Permeability'] == 0).sum(),
        (df['Permeability'] == 1).sum()
    )
    
    balanced_df = pd.concat([
        df[df['Permeability'] == 0].sample(n=min_class_size, random_state=random_state),
        df[df['Permeability'] == 1].sample(n=min_class_size, random_state=random_state)
    ]).sample(frac=1, random_state=random_state)
    
    return balanced_df

def plot_distributions(train_df, valid_df, test_df, threshold):
    """Plot permeability distributions for all datasets."""
    plt.figure(figsize=(15, 5))
    
    # Training set
    plt.subplot(1, 3, 1)
    sns.histplot(data=train_df, x='Log_Perm', bins=30)
    plt.axvline(x=np.log10(threshold), color='r', linestyle='--', label='Threshold')
    plt.title('Training Set')
    plt.xlabel('log10(Permeability) [cm/s]')
    plt.ylabel('Count')
    plt.legend()
    
    # Validation set
    plt.subplot(1, 3, 2)
    sns.histplot(data=valid_df, x='Log_Perm', bins=30)
    plt.axvline(x=np.log10(threshold), color='r', linestyle='--', label='Threshold')
    plt.title('Validation Set')
    plt.xlabel('log10(Permeability) [cm/s]')
    plt.ylabel('Count')
    plt.legend()
    
    # Test set
    plt.subplot(1, 3, 3)
    sns.histplot(data=test_df, x='Log_Perm', bins=30)
    plt.axvline(x=np.log10(threshold), color='r', linestyle='--', label='Threshold')
    plt.title('Test Set')
    plt.xlabel('log10(Permeability) [cm/s]')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs('data/analysis/model', exist_ok=True)
    plt.savefig('data/analysis/model/permeability_distributions.png')
    plt.close()

def evaluate_model(model, X, y, dataset_name):
    """Evaluate model performance."""
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y, y_pred),
        'ROC-AUC': roc_auc_score(y, y_pred_proba),
        'Precision': precision_score(y, y_pred),
        'Recall': recall_score(y, y_pred),
        'F1-score': f1_score(y, y_pred)
    }
    
    print(f"\n{dataset_name} Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
    
    return metrics

def main():
    # Load datasets
    train_df = load_and_process_data('data/raw/caco2_train.csv')
    valid_df = load_and_process_data('data/raw/caco2_valid.csv')
    test_df = load_and_process_data('data/raw/caco2_test.csv')
    
    print("\nDataset sizes:")
    print(f"Training set: {len(train_df)} compounds")
    print(f"Validation set: {len(valid_df)} compounds")
    print(f"Test set: {len(test_df)} compounds")
    
    # Create balanced datasets
    train_balanced = create_balanced_dataset(train_df)
    valid_balanced = create_balanced_dataset(valid_df)
    test_balanced = create_balanced_dataset(test_df)
    
    print("\nBalanced dataset sizes:")
    print(f"Training set: {len(train_balanced)} compounds")
    print(f"Validation set: {len(valid_balanced)} compounds")
    print(f"Test set: {len(test_balanced)} compounds")
    
    # Plot distributions
    plot_distributions(train_balanced, valid_balanced, test_balanced, threshold=8e-6)
    
    # Generate features
    print("\nGenerating Morgan fingerprints...")
    X_train = generate_morgan_fingerprints(train_balanced['SMILES'])
    X_valid = generate_morgan_fingerprints(valid_balanced['SMILES'])
    X_test = generate_morgan_fingerprints(test_balanced['SMILES'])
    
    y_train = train_balanced['Permeability']
    y_valid = valid_balanced['Permeability']
    y_test = test_balanced['Permeability']
    
    # Train model
    print("\nTraining XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='binary:logistic',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_metrics = evaluate_model(model, X_train, y_train, "Training Set")
    valid_metrics = evaluate_model(model, X_valid, y_valid, "Validation Set")
    test_metrics = evaluate_model(model, X_test, y_test, "Test Set")
    
    # Save model and metrics
    os.makedirs('data/models', exist_ok=True)
    model.save_model('data/models/xgboost_balanced.json')
    print("\nModel saved to data/models/xgboost_balanced.json")
    
    # Save metrics
    metrics = {
        'train': train_metrics,
        'valid': valid_metrics,
        'test': test_metrics
    }
    pd.DataFrame(metrics).to_json('data/models/metrics_balanced.json')
    print("Metrics saved to data/models/metrics_balanced.json")

if __name__ == '__main__':
    main()
