"""
Analyze overlap between training and ChEMBL datasets and visualize chemical space.
"""

import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import umap.umap_ as umap

def standardize_smiles(smiles):
    """Convert SMILES to a canonical form for exact matching."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

def calculate_tanimoto_similarity(mol1, mol2, radius=2):
    """Calculate Tanimoto similarity between two molecules."""
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, 2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, 2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def find_exact_duplicates(train_df, chembl_df):
    """Find exact duplicate molecules between datasets using canonical SMILES."""
    # Convert SMILES to canonical form
    train_canonical = {}
    for _, row in train_df.iterrows():
        can_smiles = standardize_smiles(row['SMILES'])
        if can_smiles:
            train_canonical[can_smiles] = row
    
    exact_matches = []
    for _, row in chembl_df.iterrows():
        can_smiles = standardize_smiles(row['SMILES'])
        if can_smiles and can_smiles in train_canonical:
            train_row = train_canonical[can_smiles]
            exact_matches.append({
                'SMILES': can_smiles,
                'Train_Name': train_row['Drug'],
                'ChEMBL_Name': row['Name'],
                'Train_Perm_cm_s': train_row['Permeability_cm_s'],
                'ChEMBL_Perm_Value': row['Permeability_Value'],
                'ChEMBL_Units': row['Original_Units'],
                'Train_Binary': train_row['Binary_Label'],
                'ChEMBL_Binary': row['Permeability']
            })
    
    return pd.DataFrame(exact_matches)

def find_similar_compounds(train_df, chembl_df, threshold=0.9):
    """Find highly similar compounds between datasets."""
    similar_pairs = []
    
    for idx1, row1 in train_df.iterrows():
        mol1 = Chem.MolFromSmiles(row1['SMILES'])
        if mol1 is None:
            continue
            
        for idx2, row2 in chembl_df.iterrows():
            mol2 = Chem.MolFromSmiles(row2['SMILES'])
            if mol2 is None:
                continue
                
            sim = calculate_tanimoto_similarity(mol1, mol2)
            if sim >= threshold:
                similar_pairs.append({
                    'Train_SMILES': row1['SMILES'],
                    'ChEMBL_SMILES': row2['SMILES'],
                    'Train_Name': row1['Drug'],
                    'ChEMBL_Name': row2['Name'],
                    'Similarity': sim,
                    'Train_Perm_cm_s': row1['Permeability_cm_s'],
                    'ChEMBL_Perm_Value': row2['Permeability_Value'],
                    'ChEMBL_Units': row2['Original_Units'],
                    'Train_Binary': row1['Binary_Label'],
                    'ChEMBL_Binary': row2['Permeability']
                })
    
    return pd.DataFrame(similar_pairs)

def plot_chemical_space_umap(train_fps, chembl_fps, train_labels, chembl_labels, save_path):
    """Create UMAP visualization of chemical space."""
    # Combine fingerprints
    all_fps = np.vstack([train_fps, chembl_fps])
    
    # Fit UMAP
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(all_fps)
    
    # Split back into train and ChEMBL
    train_umap = embedding[:len(train_fps)]
    chembl_umap = embedding[len(train_fps):]
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Training set
    plt.scatter(train_umap[train_labels==0, 0], train_umap[train_labels==0, 1],
               c='blue', alpha=0.6, label='Train (Low Perm)')
    plt.scatter(train_umap[train_labels==1, 0], train_umap[train_labels==1, 1],
               c='red', alpha=0.6, label='Train (High Perm)')
               
    # ChEMBL set
    plt.scatter(chembl_umap[chembl_labels==0, 0], chembl_umap[chembl_labels==0, 1],
               c='lightblue', alpha=0.6, label='ChEMBL (Low Perm)')
    plt.scatter(chembl_umap[chembl_labels==1, 0], chembl_umap[chembl_labels==1, 1],
               c='pink', alpha=0.6, label='ChEMBL (High Perm)')
    
    plt.title('Chemical Space (UMAP)')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def main():
    # Create output directory
    os.makedirs('data/analysis/overlap', exist_ok=True)
    
    # Load datasets
    train_df = pd.read_csv('data/caco2_train.csv')
    train_df['SMILES'] = train_df['Drug']  # Convert Drug column to SMILES
    train_df['Permeability_cm_s'] = np.power(10, train_df['Y'])  # Convert log10 to actual permeability
    train_df['Binary_Label'] = (train_df['Permeability_cm_s'] >= 8e-6).astype(int)
    
    chembl_df = pd.read_csv('data/raw/caco2_test.csv')
    
    print(f"Training set: {len(train_df)} compounds")
    print(f"ChEMBL set: {len(chembl_df)} compounds")
    
    # Analyze permeability distributions
    print("\nPermeability Distribution (Training Set):")
    print(f"Total compounds: {len(train_df)}")
    print(f"High permeability (≥8e-6 cm/s): {(train_df['Binary_Label'] == 1).sum()} ({(train_df['Binary_Label'] == 1).mean():.1%})")
    print(f"Low permeability (<8e-6 cm/s): {(train_df['Binary_Label'] == 0).sum()} ({(train_df['Binary_Label'] == 0).mean():.1%})")
    
    print("\nPermeability Distribution (ChEMBL Set):")
    print(f"Total compounds: {len(chembl_df)}")
    print(f"High permeability (≥8e-6 cm/s): {(chembl_df['Permeability'] == 1).sum()} ({(chembl_df['Permeability'] == 1).mean():.1%})")
    print(f"Low permeability (<8e-6 cm/s): {(chembl_df['Permeability'] == 0).sum()} ({(chembl_df['Permeability'] == 0).mean():.1%})")
    
    # Find exact duplicates
    exact_matches = find_exact_duplicates(train_df, chembl_df)
    exact_matches.to_csv('data/analysis/overlap/exact_matches.csv', index=False)
    
    print(f"\nFound {len(exact_matches)} exact duplicate compounds")
    if len(exact_matches) > 0:
        print("\nExample duplicates:")
        print(exact_matches.head())
        
        # Analyze agreement in exact matches
        agreement = (exact_matches['Train_Binary'] == exact_matches['ChEMBL_Binary']).mean()
        print(f"\nLabel agreement in exact matches: {agreement:.1%}")
    
    # Find similar (but not exact) compounds
    similar_pairs = find_similar_compounds(train_df, chembl_df)
    similar_pairs.to_csv('data/analysis/overlap/similar_compounds.csv', index=False)
    
    print(f"\nFound {len(similar_pairs)} similar compound pairs (Tanimoto ≥ 0.9)")
    if len(similar_pairs) > 0:
        print("\nTop 5 most similar pairs:")
        print(similar_pairs.sort_values('Similarity', ascending=False).head())
        
        # Analyze agreement in similar pairs
        agreement = (similar_pairs['Train_Binary'] == similar_pairs['ChEMBL_Binary']).mean()
        print(f"\nLabel agreement in similar pairs: {agreement:.1%}")
    
    # Generate fingerprints for UMAP
    train_fps = []
    chembl_fps = []
    
    for smiles in train_df['SMILES']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
            train_fps.append(fp)
    
    for smiles in chembl_df['SMILES']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
            chembl_fps.append(fp)
    
    train_fps = np.array(train_fps)
    chembl_fps = np.array(chembl_fps)
    
    # Get labels
    train_labels = train_df['Binary_Label']
    chembl_labels = chembl_df['Permeability'].astype(int)
    
    # Plot chemical space
    plot_chemical_space_umap(
        train_fps, chembl_fps,
        train_labels, chembl_labels,
        'data/analysis/overlap/chemical_space_umap.png'
    )
    
    print("\nResults saved to:")
    print("- data/analysis/overlap/similar_compounds.csv")
    print("- data/analysis/overlap/chemical_space_umap.png")

if __name__ == "__main__":
    main()
