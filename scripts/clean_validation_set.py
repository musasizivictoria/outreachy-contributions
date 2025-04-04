"""
Clean the ChEMBL validation set by removing duplicates and highly similar compounds.
"""
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import os

def standardize_smiles(smiles):
    """Convert SMILES to a canonical form for exact matching."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

def find_duplicates(train_df, chembl_df):
    """Find duplicate molecules between datasets using canonical SMILES."""
    # Convert training set SMILES to canonical form
    train_canonical = set()
    for smiles in train_df['SMILES']:
        can_smiles = standardize_smiles(smiles)
        if can_smiles:
            train_canonical.add(can_smiles)
    
    # Find duplicates in ChEMBL set
    duplicate_indices = []
    for idx, row in chembl_df.iterrows():
        can_smiles = standardize_smiles(row['SMILES'])
        if can_smiles in train_canonical:
            duplicate_indices.append(idx)
    
    return duplicate_indices

def calculate_similarity(mol1, mol2, radius=2):
    """Calculate Tanimoto similarity between two molecules."""
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, 2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, 2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def find_similar_compounds(train_df, chembl_df, threshold=0.9):
    """Find indices of compounds in chembl_df that are highly similar to training compounds."""
    similar_indices = set()
    
    # Convert molecules once
    train_mols = [Chem.MolFromSmiles(s) for s in train_df['SMILES']]
    train_mols = [m for m in train_mols if m is not None]
    
    for idx, row in chembl_df.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol is None:
            continue
            
        # Check similarity against all training compounds
        for train_mol in train_mols:
            similarity = calculate_similarity(mol, train_mol)
            if similarity >= threshold:
                similar_indices.add(idx)
                break
    
    return list(similar_indices)

def main():
    # Load datasets
    train_df = pd.read_csv('data/caco2_train.csv')
    chembl_df = pd.read_csv('data/raw/caco2_test.csv')
    
    # Rename Drug column to SMILES for consistency
    train_df = train_df.rename(columns={'Drug': 'SMILES'})
    
    print(f"Original ChEMBL set size: {len(chembl_df)}")
    print(f"Training set size: {len(train_df)}")
    
    # Find and remove exact duplicates
    duplicate_indices = find_duplicates(train_df, chembl_df)
    print(f"\nFound {len(duplicate_indices)} exact duplicates")
    
    # Find highly similar compounds
    similar_indices = find_similar_compounds(train_df, chembl_df)
    print(f"Found {len(similar_indices)} compounds with high similarity to training set")
    
    # Remove duplicates and similar compounds
    indices_to_remove = set(duplicate_indices + similar_indices)
    clean_df = chembl_df.drop(index=indices_to_remove)
    
    print(f"\nFinal cleaned dataset size: {len(clean_df)}")
    print("\nPermeability distribution in cleaned set:")
    print(f"High permeability (≥8e-6 cm/s): {(clean_df['Permeability'] == 1).sum()} ({(clean_df['Permeability'] == 1).mean():.1%})")
    print(f"Low permeability (<8e-6 cm/s): {(clean_df['Permeability'] == 0).sum()} ({(clean_df['Permeability'] == 0).mean():.1%})")
    
    # Save cleaned dataset
    os.makedirs('data/processed', exist_ok=True)
    clean_df.to_csv('data/processed/caco2_test_clean.csv', index=False)
    print("\nCleaned dataset saved to data/processed/caco2_test_clean.csv")

if __name__ == '__main__':
    main()
