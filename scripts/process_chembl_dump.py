"""
Process ChEMBL SQLite dump to extract recent Caco-2 permeability data.

This script extracts Caco-2 permeability data from ChEMBL database (v35)
for compounds tested between 2022-2024. It filters for standard Papp
measurements and ensures data quality through validation checks.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging
from logging.handlers import RotatingFileHandler

import pandas as pd
import sqlite3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('data/logs/chembl_processing.log', maxBytes=1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration for ChEMBL data processing."""
    db_path: Path = Path('data/raw/chembl_35/chembl_35_sqlite/chembl_35.db')
    output_path: Path = Path('data/raw/chembl_recent.csv')
    min_year: int = 2022
    assay_type: str = 'Papp'
    assay_keywords: tuple = ('caco', 'permeab')

def fetch_recent_data(config: Config) -> Optional[pd.DataFrame]:
    """Fetch recent Caco-2 data from ChEMBL SQLite dump.
    
    Args:
        config: Configuration object containing database settings
        
    Returns:
        DataFrame with Caco-2 permeability data, or None if error occurs
        
    Raises:
        sqlite3.Error: If database connection or query fails
    """
    query = """
    SELECT DISTINCT
        cs.canonical_smiles,
        a.assay_id,
        a.description,
        a.assay_type,
        d.year,
        act.standard_value,
        act.standard_units,
        act.activity_comment,
        d.chembl_id as doc_id
    FROM assays a
    JOIN activities act ON a.assay_id = act.assay_id
    JOIN compound_structures cs ON act.molregno = cs.molregno
    JOIN docs d ON a.doc_id = d.doc_id
    WHERE 
        a.description LIKE ? 
        AND a.description LIKE ?
        AND d.year >= ?
        AND act.standard_type = ?
        AND act.standard_relation = '='
        AND act.standard_value IS NOT NULL
        AND cs.canonical_smiles IS NOT NULL
    ORDER BY d.year DESC;
    """
    
    try:
        conn = sqlite3.connect(config.db_path)
        params = (
            f'%{config.assay_keywords[0]}%',
            f'%{config.assay_keywords[1]}%',
            config.min_year,
            config.assay_type
        )
        
        df = pd.read_sql_query(query, conn, params=params)
        logger.info(f"Found {len(df)} compounds from {config.min_year} onwards")
        
        return df
        
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        raise
        
    finally:
        if 'conn' in locals():
            conn.close()

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the extracted data.
    
    Args:
        df: Raw DataFrame from ChEMBL
        
    Returns:
        Cleaned DataFrame
        
    Raises:
        ValueError: If data validation fails
    """
    # Check required columns
    required_cols = ['canonical_smiles', 'standard_value', 'standard_units']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove duplicates
    df_clean = df.drop_duplicates(subset=['canonical_smiles']).copy()
    logger.info(f"Removed {len(df) - len(df_clean)} duplicate compounds")
    
    # Validate values
    df_clean = df_clean[df_clean['standard_value'] > 0]
    logger.info(f"Removed {len(df) - len(df_clean)} invalid permeability values")
    
    return df_clean

def main() -> None:
    """Main function to process ChEMBL data."""
    try:
        config = Config()
        
        # Check database exists
        if not config.db_path.exists():
            logger.error(
                "ChEMBL database not found. Please download from:\n"
                "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_35_sqlite.tar.gz\n"
                f"Extract it and place chembl_35.db at {config.db_path}"
            )
            return
        
        # Fetch and process data
        logger.info("Fetching recent data from ChEMBL...")
        df = fetch_recent_data(config)
        
        if df is not None and not df.empty:
            # Validate and clean data
            df_clean = validate_data(df)
            
            # Save to CSV
            config.output_path.parent.mkdir(parents=True, exist_ok=True)
            df_clean.to_csv(config.output_path, index=False)
            logger.info(f"Saved {len(df_clean)} compounds to {config.output_path}")
            
            # Log year distribution
            year_dist = df_clean['year'].value_counts().sort_index()
            logger.info("\nYear distribution:")
            for year, count in year_dist.items():
                logger.info(f"{year}: {count} compounds")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise
    print("\nDataset Summary:")
    print(f"Total compounds: {len(df)}")
    print("\nYears represented:")
    print(df['year'].value_counts().sort_index())
    print("\nAssay types:")
    print(df['assay_type'].value_counts())

if __name__ == '__main__':
    main()
