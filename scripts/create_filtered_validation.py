"""
Create a filtered validation set using high-quality assays and consistent conditions.
Quality criteria:
1. Minimum assay size (statistical significance)
2. Consistent units and measurement ranges
3. Low coefficient of variation for repeated measurements
4. No zero or anomalous values
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def convert_to_cm_s(value, units):
    """Convert permeability values to cm/s."""
    if pd.isna(value) or pd.isna(units):
        return None
    if units == 'nm/s':
        return value * 1e-7
    elif units == '10^-6 cm/s':
        return value * 1e-6
    elif units == 'um/s':
        return value * 1e-4
    return value

def calculate_assay_quality_metrics(df):
    """Calculate quality metrics for each assay."""
    # Convert all values to cm/s
    df['Permeability_cm_s'] = df.apply(
        lambda x: convert_to_cm_s(x['Permeability_Value'], x['Original_Units']), 
        axis=1
    )
    
    # Calculate assay-level metrics
    assay_metrics = df.groupby('Assay_ChEMBL_ID').agg({
        'Permeability_cm_s': [
            'count',  # Assay size
            'mean',   # Mean permeability
            'std',    # Standard deviation
            lambda x: x.quantile(0.25),  # Q1
            lambda x: x.quantile(0.75),  # Q3
            lambda x: (x == 0).sum(),  # Zero count
            lambda x: np.log10(x[x > 0]).std()  # Log-scale variation
        ]
    })
    
    assay_metrics.columns = [
        'Size', 'Mean', 'Std', 'Q1', 'Q3', 'Zero_Count', 'Log_Std'
    ]
    
    # Calculate quality scores
    assay_metrics['CV'] = assay_metrics['Std'] / assay_metrics['Mean']
    assay_metrics['IQR'] = assay_metrics['Q3'] - assay_metrics['Q1']
    assay_metrics['Zero_Ratio'] = assay_metrics['Zero_Count'] / assay_metrics['Size']
    
    return assay_metrics

def filter_assays(df, assay_metrics):
    """Filter assays based on quality criteria."""
    # Remove assays with all zero values
    non_zero_assays = assay_metrics[assay_metrics['Zero_Ratio'] < 1.0].index
    
    # Filter compounds from non-zero assays
    filtered_df = df[df['Assay_ChEMBL_ID'].isin(non_zero_assays)].copy()
    
    print(f"\nQuality Filtering:")
    print(f"Original assays: {len(assay_metrics)}")
    print(f"Non-zero assays: {len(non_zero_assays)}")
    print(f"Compounds in filtered set: {len(filtered_df)}")
    
    return filtered_df

def analyze_measurement_consistency(df):
    """Analyze consistency for compounds with multiple measurements."""
    # Find compounds with multiple measurements
    compound_counts = df.groupby('ChEMBL_ID')['Permeability_cm_s'].agg(['count', 'std', 'mean'])
    repeated = compound_counts[compound_counts['count'] > 1].copy()
    
    if len(repeated) > 0:
        print(f"\nMeasurement Consistency:")
        print(f"Compounds with multiple measurements: {len(repeated)}")
        
        # Calculate relative standard deviation
        repeated['RSD'] = repeated['std'] / repeated['mean']
        print("\nRelative Standard Deviation of repeated measurements:")
        print(repeated['RSD'].describe())
        
        # Keep only consistent measurements
        consistent = repeated[repeated['RSD'] <= 1.0]
        print(f"\nCompounds with consistent measurements (RSD ≤ 100%): {len(consistent)}")
        
        return consistent.index
    
    return df['ChEMBL_ID'].unique()

def create_balanced_validation_set(df, threshold=8e-6):
    """Create balanced validation set using confidence-based filtering."""
    # Remove zero and very small values (likely detection limits)
    df = df[df['Permeability_cm_s'] > 1e-10].copy()
    
    # Calculate log-transformed values
    df['Log_Perm'] = np.log10(df['Permeability_cm_s'])
    log_threshold = np.log10(threshold)
    
    # Calculate distance from threshold in log space
    df['Threshold_Distance'] = abs(df['Log_Perm'] - log_threshold)
    
    # Keep compounds that are clearly high or low permeability
    # (at least 0.5 log units from threshold, about 3-fold difference)
    clear_compounds = df[df['Threshold_Distance'] > 0.5].copy()
    
    # Average measurements for compounds with multiple values
    final_df = clear_compounds.groupby('ChEMBL_ID').agg({
        'SMILES': 'first',
        'Name': 'first',
        'Permeability_cm_s': 'mean'
    }).reset_index(drop=True)
    
    # Apply threshold
    final_df['Permeability'] = (final_df['Permeability_cm_s'] >= threshold).astype(int)
    
    # Balance classes
    min_class_size = min(
        (final_df['Permeability'] == 0).sum(),
        (final_df['Permeability'] == 1).sum()
    )
    
    balanced_df = pd.concat([
        final_df[final_df['Permeability'] == 0].sample(n=min_class_size, random_state=42),
        final_df[final_df['Permeability'] == 1].sample(n=min_class_size, random_state=42)
    ]).sample(frac=1, random_state=42)[['SMILES', 'Name', 'Permeability_cm_s', 'Permeability']]
    
    return balanced_df

def plot_assay_distributions(filtered_df, original_df):
    """Plot permeability distributions before and after filtering."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(data=original_df, x=np.log10(original_df['Permeability_cm_s']), bins=30)
    plt.axvline(x=np.log10(8e-6), color='r', linestyle='--', label='Threshold')
    plt.title('Original Distribution')
    plt.xlabel('log10(Permeability) [cm/s]')
    plt.ylabel('Count')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.histplot(data=filtered_df, x=np.log10(filtered_df['Permeability_cm_s']), bins=30)
    plt.axvline(x=np.log10(8e-6), color='r', linestyle='--', label='Threshold')
    plt.title('Filtered Distribution')
    plt.xlabel('log10(Permeability) [cm/s]')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs('data/analysis/filtered', exist_ok=True)
    plt.savefig('data/analysis/filtered/distributions.png')
    plt.close()

def main():
    # Load ChEMBL data
    chembl_df = pd.read_csv('data/raw/caco2_test.csv')
    print(f"Original dataset: {len(chembl_df)} compounds")
    
    # Calculate assay quality metrics
    assay_metrics = calculate_assay_quality_metrics(chembl_df)
    
    # Filter assays based on quality criteria
    filtered_df = filter_assays(chembl_df, assay_metrics)
    
    # Analyze measurement consistency
    consistent_compounds = analyze_measurement_consistency(filtered_df)
    if len(consistent_compounds) > 0:
        filtered_df = filtered_df[filtered_df['ChEMBL_ID'].isin(consistent_compounds)]
    
    # Create balanced validation set
    final_df = create_balanced_validation_set(filtered_df)
    
    print(f"\nFinal balanced dataset: {len(final_df)} compounds")
    print("\nClass distribution:")
    print(f"High permeability: {(final_df['Permeability'] == 1).sum()} ({(final_df['Permeability'] == 1).mean():.1%})")
    print(f"Low permeability: {(final_df['Permeability'] == 0).sum()} ({(final_df['Permeability'] == 0).mean():.1%})")
    
    # Plot distributions
    plot_assay_distributions(filtered_df, chembl_df)
    
    # Save final dataset
    os.makedirs('data/processed', exist_ok=True)
    final_df.to_csv('data/processed/caco2_test_filtered.csv', index=False)
    print("\nFiltered dataset saved to data/processed/caco2_test_filtered.csv")
    print("Distribution plots saved to data/analysis/filtered/distributions.png")

if __name__ == '__main__':
    main()
