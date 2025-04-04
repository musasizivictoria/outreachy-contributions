"""
Analyze permeability value distributions and binarization thresholds across datasets.
"""
import pandas as pd
import numpy as np
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

def analyze_training_set():
    """Analyze training set permeability distribution."""
    train_df = pd.read_csv('data/caco2_train.csv')
    # Y values are already in log10(cm/s)
    train_df['Permeability_cm_s'] = 10 ** train_df['Y']
    train_df['Binary'] = (train_df['Permeability_cm_s'] >= 8e-6).astype(int)
    
    return train_df

def analyze_chembl_set():
    """Analyze ChEMBL set permeability distribution."""
    chembl_df = pd.read_csv('data/raw/caco2_test.csv')
    chembl_df['Permeability_cm_s'] = chembl_df.apply(
        lambda x: convert_to_cm_s(x['Permeability_Value'], x['Original_Units']), 
        axis=1
    )
    return chembl_df

def plot_distributions(train_df, chembl_df):
    """Plot permeability distributions."""
    plt.figure(figsize=(12, 6))
    
    # Plot training set distribution
    plt.subplot(1, 2, 1)
    sns.histplot(data=train_df, x=np.log10(train_df['Permeability_cm_s']), bins=30)
    plt.axvline(x=np.log10(8e-6), color='r', linestyle='--', label='Threshold (8e-6 cm/s)')
    plt.title('Training Set Distribution')
    plt.xlabel('log10(Permeability) [cm/s]')
    plt.ylabel('Count')
    plt.legend()
    
    # Plot ChEMBL set distribution
    plt.subplot(1, 2, 2)
    sns.histplot(data=chembl_df, x=np.log10(chembl_df['Permeability_cm_s']), bins=30)
    plt.axvline(x=np.log10(8e-6), color='r', linestyle='--', label='Threshold (8e-6 cm/s)')
    plt.title('ChEMBL Set Distribution')
    plt.xlabel('log10(Permeability) [cm/s]')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs('data/analysis/permeability', exist_ok=True)
    plt.savefig('data/analysis/permeability/distributions.png')
    plt.close()

def analyze_value_ranges(df, name):
    """Analyze permeability value ranges."""
    print(f"\n{name} Permeability Analysis:")
    print(f"Total compounds: {len(df)}")
    print(f"Compounds with permeability values: {df['Permeability_cm_s'].notna().sum()}")
    print("\nPermeability ranges (cm/s):")
    
    ranges = [
        ('< 0.1 µm/s', lambda x: x < 1e-7),
        ('0.1-1 µm/s', lambda x: (x >= 1e-7) & (x < 1e-6)),
        ('1-10 µm/s', lambda x: (x >= 1e-6) & (x < 1e-5)),
        ('10-100 µm/s', lambda x: (x >= 1e-5) & (x < 1e-4)),
        ('> 100 µm/s', lambda x: x >= 1e-4)
    ]
    
    for range_name, range_func in ranges:
        count = df[range_func(df['Permeability_cm_s'])].shape[0]
        print(f"{range_name}: {count} ({count/len(df):.1%})")
    
    print("\nSummary statistics (cm/s):")
    print(df['Permeability_cm_s'].describe())

def main():
    # Analyze training set
    train_df = analyze_training_set()
    analyze_value_ranges(train_df, "Training Set")
    
    # Analyze ChEMBL set
    chembl_df = analyze_chembl_set()
    analyze_value_ranges(chembl_df, "ChEMBL Set")
    
    # Plot distributions
    plot_distributions(train_df, chembl_df)
    
    print("\nResults saved to data/analysis/permeability/distributions.png")

if __name__ == '__main__':
    main()
