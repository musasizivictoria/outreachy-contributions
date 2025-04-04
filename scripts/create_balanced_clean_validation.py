"""
Create a balanced validation set from the cleaned ChEMBL data.
"""
import pandas as pd
import numpy as np

def create_balanced_dataset(df, target_col='Permeability', random_state=42):
    """Create a balanced dataset by randomly sampling from each class."""
    # Get counts for each class
    class_counts = df[target_col].value_counts()
    min_class_size = class_counts.min()
    
    # Sample equally from each class
    balanced_samples = []
    for class_label in class_counts.index:
        class_data = df[df[target_col] == class_label]
        sampled = class_data.sample(n=min_class_size, random_state=random_state)
        balanced_samples.append(sampled)
    
    # Combine and shuffle
    balanced_df = pd.concat(balanced_samples, ignore_index=True)
    return balanced_df.sample(frac=1, random_state=random_state)

def main():
    # Load cleaned dataset
    df = pd.read_csv('data/processed/caco2_test_clean.csv')
    print(f"\nOriginal cleaned dataset size: {len(df)}")
    print("\nOriginal class distribution:")
    print(f"High permeability (≥8e-6 cm/s): {(df['Permeability'] == 1).sum()} ({(df['Permeability'] == 1).mean():.1%})")
    print(f"Low permeability (<8e-6 cm/s): {(df['Permeability'] == 0).sum()} ({(df['Permeability'] == 0).mean():.1%})")
    
    # Create balanced dataset
    balanced_df = create_balanced_dataset(df)
    print(f"\nBalanced dataset size: {len(balanced_df)}")
    print("\nBalanced class distribution:")
    print(f"High permeability (≥8e-6 cm/s): {(balanced_df['Permeability'] == 1).sum()} ({(balanced_df['Permeability'] == 1).mean():.1%})")
    print(f"Low permeability (<8e-6 cm/s): {(balanced_df['Permeability'] == 0).sum()} ({(balanced_df['Permeability'] == 0).mean():.1%})")
    
    # Save balanced dataset
    balanced_df.to_csv('data/processed/caco2_test_clean_balanced.csv', index=False)
    print("\nBalanced dataset saved to data/processed/caco2_test_clean_balanced.csv")

if __name__ == '__main__':
    main()
