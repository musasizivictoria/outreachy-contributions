"""
Create an improved validation set by:
1. Removing compounds with zero permeability
2. Log-transforming values for better comparison
3. Adjusting threshold based on ChEMBL distribution
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

def convert_to_cm_s(value, units):
    """Convert permeability values to cm/s."""
    if pd.isna(value) or pd.isna(units):
        return None
    if units == 'nm/s':
        return value * 1e-7
    elif units == '10^-6 cm/s':
        return value * 1e-6
    return value

def log_transform_permeability(value):
    """Log transform permeability with handling for zeros/negatives."""
    if value <= 0:
        return None
    return np.log10(value)

def find_optimal_threshold(train_values, chembl_values):
    """Find optimal threshold that maintains similar class proportions."""
    # Convert to log space
    train_log = np.log10(train_values)
    chembl_log = np.log10(chembl_values)
    
    # Get training set threshold in log space
    train_threshold = np.log10(8e-6)
    
    # Calculate class proportions in training set
    train_high = (train_log >= train_threshold).mean()
    
    # Find threshold in ChEMBL set that gives similar proportions
    percentile = (1 - train_high) * 100
    chembl_threshold = np.percentile(chembl_log, percentile)
    
    return 10 ** chembl_threshold

def create_improved_validation_set():
    """Create improved validation set with better threshold."""
    # Load datasets
    train_df = pd.read_csv('data/caco2_train.csv')
    chembl_df = pd.read_csv('data/raw/caco2_test.csv')
    
    # Process training set
    train_df['Permeability_cm_s'] = 10 ** train_df['Y']
    train_df = train_df[train_df['Permeability_cm_s'] > 0]
    
    # Process ChEMBL set
    chembl_df['Permeability_cm_s'] = chembl_df.apply(
        lambda x: convert_to_cm_s(x['Permeability_Value'], x['Original_Units']), 
        axis=1
    )
    
    # Remove zeros and invalid values
    chembl_df = chembl_df[chembl_df['Permeability_cm_s'] > 0]
    
    # Find optimal threshold
    threshold = find_optimal_threshold(
        train_df['Permeability_cm_s'].values,
        chembl_df['Permeability_cm_s'].values
    )
    
    # Apply threshold
    chembl_df['Permeability'] = (chembl_df['Permeability_cm_s'] >= threshold).astype(int)
    
    # Create balanced dataset
    min_class_size = min(
        (chembl_df['Permeability'] == 0).sum(),
        (chembl_df['Permeability'] == 1).sum()
    )
    
    balanced_df = pd.concat([
        chembl_df[chembl_df['Permeability'] == 0].sample(n=min_class_size, random_state=42),
        chembl_df[chembl_df['Permeability'] == 1].sample(n=min_class_size, random_state=42)
    ]).sample(frac=1, random_state=42)
    
    # Plot distributions
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(data=train_df, x=np.log10(train_df['Permeability_cm_s']), bins=30, label='Training')
    plt.axvline(x=np.log10(8e-6), color='r', linestyle='--', label='Original threshold')
    plt.title('Training Set Distribution')
    plt.xlabel('log10(Permeability) [cm/s]')
    plt.ylabel('Count')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.histplot(data=balanced_df, x=np.log10(balanced_df['Permeability_cm_s']), bins=30, label='ChEMBL')
    plt.axvline(x=np.log10(threshold), color='r', linestyle='--', label=f'Adjusted threshold')
    plt.title('Balanced ChEMBL Set Distribution')
    plt.xlabel('log10(Permeability) [cm/s]')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs('data/analysis/permeability', exist_ok=True)
    plt.savefig('data/analysis/permeability/improved_distributions.png')
    plt.close()
    
    return balanced_df, threshold

def main():
    # Create improved validation set
    improved_df, threshold = create_improved_validation_set()
    
    print("\nImproved Validation Set:")
    print(f"Total compounds: {len(improved_df)}")
    print(f"Compounds removed due to zero/invalid permeability: {991 - len(improved_df)}")
    print(f"\nAdjusted threshold: {threshold:.2e} cm/s")
    print("\nClass distribution:")
    print(f"High permeability: {(improved_df['Permeability'] == 1).sum()} ({(improved_df['Permeability'] == 1).mean():.1%})")
    print(f"Low permeability: {(improved_df['Permeability'] == 0).sum()} ({(improved_df['Permeability'] == 0).mean():.1%})")
    
    # Save improved dataset
    os.makedirs('data/processed', exist_ok=True)
    improved_df.to_csv('data/processed/caco2_test_improved.csv', index=False)
    print("\nImproved dataset saved to data/processed/caco2_test_improved.csv")
    print("Distribution plots saved to data/analysis/permeability/improved_distributions.png")

if __name__ == '__main__':
    main()
