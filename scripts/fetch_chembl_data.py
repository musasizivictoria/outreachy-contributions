"""
Fetch Caco-2 permeability data from ChEMBL database.

This script fetches Caco-2 permeability data from ChEMBL's web API,
processes and standardizes the measurements, and creates train/valid/test
splits for model development.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging
from logging.handlers import RotatingFileHandler

import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('data/logs/chembl_fetch.log', maxBytes=1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration for ChEMBL data fetching and processing."""
    output_dir: Path = Path('data/external_validation')
    raw_dir: Path = Path('data/raw')
    permeability_threshold: float = 8e-6  # cm/s
    train_fraction: float = 0.7
    valid_fraction: float = 0.15
    random_seed: int = 42
    standard_types: tuple = ('Papp', 'Permeability')
    standard_units: tuple = ('cm/s', 'nm/s', 'um/s', '10^-6 cm/s')
    required_fields: tuple = (
        'standard_value', 'standard_units', 
        'canonical_smiles', 'molecule_chembl_id'
    )

def fetch_caco2_data(config: Config) -> List[Dict[str, Any]]:
    """Fetch Caco-2 permeability data directly from activities.
    
    Args:
        config: Configuration object containing API filters
        
    Returns:
        List of activity records from ChEMBL
        
    Raises:
        Exception: If API request fails
    """
    try:
        activity = new_client.activity
        
        # Search for Caco-2 permeability measurements
        results = activity.filter(
            assay_desc__icontains="caco-2"
        ).filter(
            standard_type__in=config.standard_types
        ).filter(
            standard_units__in=config.standard_units
        )
        
        data = list(results)
        logger.info(f"Retrieved {len(data)} measurements from ChEMBL")
        return data
        
    except Exception as e:
        logger.error(f"Failed to fetch data from ChEMBL: {e}")
        raise

def process_permeability_value(
    value: float, 
    units: str,
    relation: str,
    threshold: float
) -> Tuple[float, bool]:
    """Process and standardize permeability values.
    
    Args:
        value: Raw permeability value
        units: Original measurement units
        relation: Comparison operator ('=', '>', '<')
        threshold: Permeability threshold in cm/s
        
    Returns:
        Tuple of (standardized value in cm/s, binary permeability class)
        
    Raises:
        ValueError: If units are not recognized
    """
    # Convert to cm/s if needed
    conversion_factors = {
        'nm/s': 1e-7,
        'um/s': 1e-4,
        '10^-6 cm/s': 1e-6,
        'cm/s': 1.0
    }
    
    if units not in conversion_factors:
        raise ValueError(f"Unrecognized units: {units}")
        
    value = value * conversion_factors[units]
    
    # Handle relations
    relation_factors = {
        '>': 1.1,  # Conservative estimate
        '<': 0.9,
        '=': 1.0
    }
    
    if relation not in relation_factors:
        logger.warning(f"Unknown relation '{relation}', treating as '='")
        relation = '='
        
    value = value * relation_factors[relation]
    
    # Convert to binary class
    is_permeable = value >= threshold
    
    return value, is_permeable

def process_results(results: List[Dict[str, Any]], config: Config) -> pd.DataFrame:
    """Process raw ChEMBL results into a clean DataFrame.
    
    Args:
        results: Raw results from ChEMBL API
        config: Configuration object
        
    Returns:
        DataFrame with processed permeability data
    """
    data = []
    error_count = 0
    
    for result in results:
        try:
            # Check required fields
            if not all(result.get(k) for k in config.required_fields):
                continue
            
            # Get relation, default to '=' if not specified
            relation = result.get('standard_relation', '=')
            
            try:
                value, is_permeable = process_permeability_value(
                    float(result['standard_value']),
                    result['standard_units'],
                    relation,
                    config.permeability_threshold
                )
                
                data.append({
                    'SMILES': result['canonical_smiles'],
                    'Name': result.get('molecule_pref_name', 'Unknown'),
                    'ChEMBL_ID': result['molecule_chembl_id'],
                    'Assay_ChEMBL_ID': result.get('assay_chembl_id', 'Unknown'),
                    'Permeability_Value': value,
                    'Original_Units': result['standard_units'],
                    'Permeability': int(is_permeable)
                })
                
            except Exception as e:
                logger.warning(f"Error processing value: {e}")
                error_count += 1
                continue
                
        except Exception as e:
            logger.warning(f"Error processing result: {e}")
            error_count += 1
            continue
    
    if error_count > 0:
        logger.warning(f"Failed to process {error_count} results")
        
    if not data:
        logger.error("No valid data found")
        return pd.DataFrame()
    
    return pd.DataFrame(data)

def aggregate_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate duplicate compounds by taking median values.
    
    Args:
        df: Raw DataFrame with possible duplicates
        
    Returns:
        DataFrame with one row per unique compound
    """
    if df.empty:
        return df
        
    df_median = df.groupby('SMILES').agg({
        'Name': 'first',
        'ChEMBL_ID': 'first',
        'Permeability_Value': 'median',
        'Permeability': lambda x: int(x.mode()[0])
    }).reset_index()
    
    logger.info(f"Aggregated {len(df)} measurements into {len(df_median)} unique compounds")
    return df_median

def create_data_splits(df: pd.DataFrame, config: Config) -> None:
    """Create and save train/valid/test splits.
    
    Args:
        df: Clean DataFrame to split
        config: Configuration object with split ratios
    """
    if df.empty:
        logger.warning("No data to split")
        return
        
    # Shuffle data
    df = df.sample(frac=1, random_state=config.random_seed).reset_index(drop=True)
    
    # Calculate split sizes
    train_size = int(config.train_fraction * len(df))
    valid_size = int(config.valid_fraction * len(df))
    
    # Split data
    train_df = df[:train_size]
    valid_df = df[train_size:train_size + valid_size]
    test_df = df[train_size + valid_size:]
    
    # Save splits
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(config.raw_dir / 'caco2_train.csv', index=False)
    valid_df.to_csv(config.raw_dir / 'caco2_valid.csv', index=False)
    test_df.to_csv(config.raw_dir / 'caco2_test.csv', index=False)
    
    logger.info(f"\nSplit data into:")
    logger.info(f"Train: {len(train_df)} compounds")
    logger.info(f"Valid: {len(valid_df)} compounds")
    logger.info(f"Test: {len(test_df)} compounds")

def fetch_and_process_data() -> Optional[pd.DataFrame]:
    """Fetch and process Caco-2 data from ChEMBL.
    
    Returns:
        Processed DataFrame or None if error occurs
    """
    try:
        config = Config()
        
        # Fetch data
        logger.info("Fetching Caco-2 permeability data from ChEMBL...")
        results = fetch_caco2_data(config)
        
        # Process results
        df = process_results(results, config)
        if df.empty:
            return None
            
        # Aggregate duplicates
        df_clean = aggregate_duplicates(df)
        
        # Save full dataset
        config.output_dir.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(config.output_dir / 'chembl_caco2.csv', index=False)
        logger.info(f"Saved {len(df_clean)} compounds to {config.output_dir / 'chembl_caco2.csv'}")
        
        # Create splits
        create_data_splits(df_clean, config)
        
        return df_clean
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return None
    
    data = []
    for result in results:
        try:
            # Check required fields
            required_fields = ['standard_value', 'standard_units', 'canonical_smiles', 'molecule_chembl_id']
            if not all(result.get(k) for k in required_fields):
                continue
            
            # Get relation, default to '=' if not specified
            relation = result.get('standard_relation', '=')
            
            try:
                value, is_permeable = process_permeability_value(
                    float(result['standard_value']),
                    result['standard_units'],
                    relation
                )
                
                data.append({
                    'SMILES': result['canonical_smiles'],
                    'Name': result.get('molecule_pref_name', 'Unknown'),
                    'ChEMBL_ID': result['molecule_chembl_id'],
                    'Assay_ChEMBL_ID': result.get('assay_chembl_id', 'Unknown'),
                    'Permeability_Value': value,
                    'Original_Units': result['standard_units'],
                    'Permeability': int(is_permeable)
                })
            except Exception as e:
                print(f"Error processing value: {e}")
                continue
                
        except Exception as e:
            print(f"Error processing result: {e}")
            continue
    
    if not data:
        print("No valid data found")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    print(f"Found {len(df)} valid compounds")
    
    # Remove duplicates, keeping median value for each compound
    if len(df) > 0:
        df_median = df.groupby('SMILES').agg({
            'Name': 'first',
            'ChEMBL_ID': 'first',
            'Permeability_Value': 'median',
            'Permeability': lambda x: int(x.mode()[0])
        }).reset_index()
        
        print(f"After aggregating duplicates: {len(df_median)} unique compounds")
    
    # Save the data
    os.makedirs('data/external_validation', exist_ok=True)
    df.to_csv('data/external_validation/chembl_caco2.csv', index=False)
    print(f"Saved {len(df)} compounds to data/external_validation/chembl_caco2.csv")
    
    # Create train/valid/test splits
    os.makedirs('data/raw', exist_ok=True)
    
    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split data
    train_size = int(0.7 * len(df))
    valid_size = int(0.15 * len(df))
    
    train_df = df[:train_size]
    valid_df = df[train_size:train_size + valid_size]
    test_df = df[train_size + valid_size:]
    
    # Save splits
    train_df.to_csv('data/raw/caco2_train.csv', index=False)
    valid_df.to_csv('data/raw/caco2_valid.csv', index=False)
    test_df.to_csv('data/raw/caco2_test.csv', index=False)
    
    print(f"\nSplit data into:")
    print(f"Train: {len(train_df)} compounds")
    print(f"Valid: {len(valid_df)} compounds")
    print(f"Test: {len(test_df)} compounds")
    
    return df

if __name__ == "__main__":
    fetch_and_process_data()
