"""
Analyze chemical space coverage of training and external validation sets.
"""

import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple

def calculate_descriptors(smiles: str) -> Dict[str, float]:
    """Calculate key molecular descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
        
    return {
        'MW': Descriptors.ExactMolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'RotBonds': Descriptors.NumRotatableBonds(mol)
    }

def analyze_dataset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Calculate descriptors for a dataset."""
    results = []
    for _, row in df.iterrows():
        desc = calculate_descriptors(row['SMILES'])
        if desc is not None:
            desc['Dataset'] = name
            desc['Permeability'] = row['Permeability']
            results.append(desc)
    
    return pd.DataFrame(results)

def plot_chemical_space(
    train_desc: pd.DataFrame,
    external_desc: pd.DataFrame,
    save_dir: str
):
    """Create chemical space visualization plots."""
    # Combine datasets
    all_desc = pd.concat([train_desc, external_desc])
    
    # PCA on molecular descriptors
    X = all_desc[['MW', 'LogP', 'TPSA', 'HBA', 'HBD', 'RotBonds']]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Create plots directory
    os.makedirs(save_dir, exist_ok=True)
    
    # PCA plot
    plt.figure(figsize=(10, 8))
    for dataset in ['Training', 'External']:
        mask = all_desc['Dataset'] == dataset
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            alpha=0.6,
            label=dataset
        )
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('Chemical Space Coverage (PCA)')
    plt.legend()
    plt.savefig(f'{save_dir}/chemical_space_pca.png')
    plt.close()
    
    # Property distributions
    for prop in ['MW', 'LogP', 'TPSA']:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(
            data=all_desc,
            x=prop,
            hue='Dataset',
            common_norm=False
        )
        plt.title(f'{prop} Distribution')
        plt.savefig(f'{save_dir}/{prop}_distribution.png')
        plt.close()
    
    # Summary statistics
    summary = all_desc.groupby('Dataset').agg({
        'MW': ['mean', 'std', 'min', 'max'],
        'LogP': ['mean', 'std', 'min', 'max'],
        'TPSA': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    summary.to_csv(f'{save_dir}/descriptor_summary.csv')

def main():
    # Load data
    train_data = pd.read_csv('data/raw/caco2_train.csv')
    external_data = pd.read_csv('data/external_validation/chembl_caco2.csv')
    
    print(f"Analyzing chemical space for {len(train_data)} training compounds and {len(external_data)} external compounds...")
    
    # Calculate descriptors
    train_desc = analyze_dataset(train_data, 'Training')
    external_desc = analyze_dataset(external_data, 'External')
    
    print(f"\nDescriptor calculation complete:")
    print(f"Training set: {len(train_desc)} valid compounds")
    print(f"External set: {len(external_desc)} valid compounds")
    
    # Plot chemical space
    save_dir = 'data/analysis/chemical_space'
    plot_chemical_space(train_desc, external_desc, save_dir)
    
    print(f"\nAnalysis complete. Results saved to {save_dir}/")
    
    # Load external validation data
    external_data = pd.read_csv('data/external_validation/chembl_caco2.csv')
    
    # Calculate descriptors
    train_desc = analyze_dataset(train_data, 'Training')
    external_desc = analyze_dataset(external_data, 'External')
    
    # Plot and analyze
    plot_chemical_space(
        train_desc,
        external_desc,
        'data/external_validation/chemical_space'
    )
    
    print("Chemical space analysis complete. Results saved to data/external_validation/chemical_space/")

if __name__ == "__main__":
    main()
