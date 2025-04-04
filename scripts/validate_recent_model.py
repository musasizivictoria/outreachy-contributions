"""
Validate our balanced model on recent ChEMBL data (2022-2024).

This script performs validation of the trained XGBoost model on recent ChEMBL data,
including feature generation, prediction, and performance evaluation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Union, Optional
import json
import logging
from logging.handlers import RotatingFileHandler

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('data/logs/validation.log', maxBytes=1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
PERMEABILITY_THRESHOLD = 8e-6  # cm/s
FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048

@dataclass
class Config:
    """Configuration for model validation."""
    data_path: Path = Path('data/raw/chembl_recent.csv')
    model_path: Path = Path('data/models/xgboost_balanced.json')
    output_dir: Path = Path('data/analysis')
    metrics_path: Path = Path('data/analysis/recent_chembl_metrics.json')
    plot_path: Path = Path('data/analysis/recent_permeability_dist.png')
    threshold: float = PERMEABILITY_THRESHOLD
    fingerprint_radius: int = FINGERPRINT_RADIUS
    fingerprint_bits: int = FINGERPRINT_BITS

def convert_to_cm_s(row: pd.Series) -> float:
    """Convert permeability values to cm/s.
    
    Args:
        row: DataFrame row containing 'standard_value' and 'standard_units'
        
    Returns:
        float: Permeability value in cm/s
        
    Raises:
        ValueError: If units are not recognized
    """
    value = row['standard_value']
    units = row['standard_units']
    
    conversion_factors = {
        'nm/s': 1e-7,
        '10^-6 cm/s': 1e-6,
        'um/s': 1e-4,
        'cm/s': 1.0
    }
    
    if units not in conversion_factors:
        raise ValueError(f"Unrecognized units: {units}")
        
    return value * conversion_factors[units]

def generate_morgan_fingerprints(smiles_list: List[str], radius: int = FINGERPRINT_RADIUS, 
                              nBits: int = FINGERPRINT_BITS) -> np.ndarray:
    """Generate Morgan fingerprints for a list of SMILES.
    
    Args:
        smiles_list: List of SMILES strings
        radius: Morgan fingerprint radius
        nBits: Number of bits in fingerprint
        
    Returns:
        np.ndarray: Array of fingerprints
        
    Raises:
        ValueError: If any SMILES string is invalid
    """
    fingerprints = []
    error_count = 0
    
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
            fingerprints.append(list(fp.ToBitString()))
            
        except Exception as e:
            logger.warning(f"Error processing SMILES {smiles}: {e}")
            fingerprints.append([0] * nBits)
            error_count += 1
    
    if error_count > 0:
        logger.warning(f"Failed to process {error_count} SMILES strings")
        
    return np.array(fingerprints, dtype=int)

def plot_permeability_distribution(df: pd.DataFrame, threshold: float, output_file: Union[str, Path]) -> None:
    """Plot permeability value distribution.
    
    Args:
        df: DataFrame with 'Log_Perm' column
        threshold: Permeability threshold in cm/s
        output_file: Path to save the plot
        
    Raises:
        ValueError: If required columns are missing
    """
    if 'Log_Perm' not in df.columns:
        raise ValueError("DataFrame must contain 'Log_Perm' column")
        
    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    sns.histplot(data=df, x='Log_Perm', bins=30)
    plt.axvline(x=np.log10(threshold), color='r', linestyle='--', label='Threshold')
    
    plt.title('Permeability Distribution (Recent ChEMBL Data)')
    plt.xlabel('log10(Permeability) [cm/s]')
    plt.ylabel('Count')
    plt.legend()
    
    # Save plot
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def validate_model(config: Config) -> Dict[str, float]:
    """Validate model on recent ChEMBL data.
    
    Args:
        config: Configuration object
        
    Returns:
        dict: Model performance metrics
        
    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If data processing fails
    """
    logger.info("Starting model validation")
    
    # Load and process data
    logger.info(f"Loading data from {config.data_path}")
    try:
        df = pd.read_csv(config.data_path)
    except FileNotFoundError:
        logger.error(f"Data file not found: {config.data_path}")
        raise
    
    # Convert and validate permeability values
    logger.info("Processing permeability values")
    try:
        df['Permeability_cm_s'] = df.apply(convert_to_cm_s, axis=1)
        df = df[df['Permeability_cm_s'] > 0].copy()
        
        if len(df) == 0:
            raise ValueError("No valid permeability values found")
            
        df['Log_Perm'] = np.log10(df['Permeability_cm_s'])
        df['Permeability'] = (df['Permeability_cm_s'] >= config.threshold).astype(int)
        
    except Exception as e:
        logger.error(f"Error processing permeability values: {e}")
        raise
    
    # Log dataset statistics
    logger.info(f"Processed dataset: {len(df)} compounds")
    high_perm = (df['Permeability'] == 1).sum()
    low_perm = (df['Permeability'] == 0).sum()
    logger.info(f"Class distribution: {high_perm} high, {low_perm} low")
    
    # Generate plot
    logger.info(f"Generating distribution plot at {config.plot_path}")
    plot_permeability_distribution(df, config.threshold, config.plot_path)
    
    # Generate features
    logger.info("Generating Morgan fingerprints")
    X = generate_morgan_fingerprints(
        df['canonical_smiles'],
        radius=config.fingerprint_radius,
        nBits=config.fingerprint_bits
    )
    y = df['Permeability']
    
    # Load and apply model
    logger.info(f"Loading model from {config.model_path}")
    try:
        model = xgb.XGBClassifier()
        model.load_model(str(config.model_path))
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    # Make predictions
    logger.info("Making predictions")
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Calculate metrics
    metrics = {
        'Accuracy': float(accuracy_score(y, y_pred)),
        'ROC-AUC': float(roc_auc_score(y, y_pred_proba)),
        'Precision': float(precision_score(y, y_pred)),
        'Recall': float(recall_score(y, y_pred)),
        'F1-score': float(f1_score(y, y_pred))
    }
    
    # Save metrics
    logger.info(f"Saving metrics to {config.metrics_path}")
    config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def main() -> None:
    """Main function to run model validation."""
    try:
        config = Config()
        metrics = validate_model(config)
        
        logger.info("\nValidation Results:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.3f}")
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise

if __name__ == '__main__':
    main()
    
    # Analyze predictions by permeability range
    df['Predicted'] = y_pred
    df['Correct'] = (df['Permeability'] == df['Predicted'])
    
    # Define permeability ranges
    def get_range(x):
        if x < 1e-7:
            return '< 0.1 µm/s'
        elif x < 1e-6:
            return '0.1-1 µm/s'
        elif x < 1e-5:
            return '1-10 µm/s'
        elif x < 1e-4:
            return '10-100 µm/s'
        else:
            return '> 100 µm/s'
    
    df['Perm_Range'] = df['Permeability_cm_s'].apply(get_range)
    
    print("\nPrediction Analysis by Permeability Range:")
    range_analysis = df.groupby('Perm_Range').agg({
        'Correct': ['count', 'mean']
    })
    range_analysis.columns = ['Count', 'Accuracy']
    print(range_analysis)
    
    # Save results
    os.makedirs('data/analysis', exist_ok=True)
    pd.DataFrame([metrics]).to_json('data/analysis/recent_chembl_metrics.json')
    print("\nResults saved to data/analysis/recent_chembl_metrics.json")

if __name__ == '__main__':
    main()
