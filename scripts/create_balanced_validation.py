"""
Create a balanced validation set from ChEMBL data by normalizing permeability values.
"""

import pandas as pd
import numpy as np

def convert_to_cm_per_s(value, unit):
    """Convert permeability values to cm/s."""
    if pd.isna(unit) or pd.isna(value):
        return None
        
    if unit == '10^-6 cm/s':
        return value * 1e-6
    elif unit == 'nm/s':
        return value * 1e-7  # 1 nm/s = 1e-7 cm/s
    elif unit == 'cm/s':
        return value
    elif unit == 'um/s':
        return value * 1e-4  # 1 µm/s = 1e-4 cm/s
    else:
        print(f"Warning: Unknown unit '{unit}' for value {value}")
        return None

def main():
    # Load ChEMBL data
    data = pd.read_csv('data/raw/caco2_test.csv')
    print(f"Loaded {len(data)} compounds")
    
    # Convert permeability values to cm/s
    data['Permeability_cm_s'] = [
        convert_to_cm_per_s(val, unit) 
        for val, unit in zip(data['Permeability_Value'], data['Original_Units'])
    ]
    
    # Remove rows with invalid permeability values
    valid_data = data.dropna(subset=['Permeability_cm_s']).copy()
    print(f"Found {len(valid_data)} compounds with valid permeability values")
    
    # Use same threshold as training data: 8e-6 cm/s
    threshold = 8e-6
    valid_data['Calculated_Binary'] = (valid_data['Permeability_cm_s'] >= threshold).astype(int)
    
    # Compare with provided binary labels
    agreement = (valid_data['Calculated_Binary'] == valid_data['Permeability']).mean()
    print(f"\nAgreement with provided binary labels: {agreement:.1%}")
    
    # Print class distribution
    print("\nClass distribution before balancing:")
    print(valid_data['Calculated_Binary'].value_counts(normalize=True))
    
    # Print permeability value ranges
    print("\nPermeability ranges (cm/s):")
    print(valid_data.groupby('Calculated_Binary')['Permeability_cm_s'].describe())
    
    # Create balanced dataset
    class_0 = valid_data[valid_data['Calculated_Binary'] == 0]
    class_1 = valid_data[valid_data['Calculated_Binary'] == 1]
    
    min_size = min(len(class_0), len(class_1))
    balanced_data = pd.concat([
        class_0.sample(min_size, random_state=42),
        class_1.sample(min_size, random_state=42)
    ])
    
    # Sort by permeability value for easier inspection
    balanced_data = balanced_data.sort_values('Permeability_cm_s')
    
    print(f"\nBalanced dataset size: {len(balanced_data)} compounds")
    print("Class distribution after balancing:")
    print(balanced_data['Calculated_Binary'].value_counts(normalize=True))
    
    # Save balanced dataset
    balanced_data.to_csv('data/external_validation/balanced_validation.csv', index=False)
    print("\nSaved balanced dataset to data/external_validation/balanced_validation.csv")

if __name__ == "__main__":
    main()
