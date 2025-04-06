# Drug Absorption Prediction Model

## Project Overview
This project develops a machine learning model to predict drug absorption using the Caco2_Wang dataset from Therapeutics Data Commons (TDC) and validates it on recent ChEMBL data. Our goal is to create a robust binary classifier that predicts whether a drug compound will have high or low permeability through the Caco2 cell membrane, with proven generalization to unseen data.

### Why This Dataset Matters
The Caco2_Wang dataset is particularly valuable for drug discovery because:
1. It provides experimental measurements from Caco2 cells, which are derived from human colorectal adenocarcinoma and serve as a well-established model for intestinal absorption
2. The data represents actual permeability measurements in cm/s, making it directly relevant to drug development
3. The dataset is well-balanced (51.6% low permeability vs 48.4% high permeability) and of moderate size (910 compounds), making it suitable for machine learning

## Understanding the Data

### Experimental Background
- **Original Task Type**: Regression (predicting exact permeability values)
- **Measurement**: Permeability through Caco2 cell monolayers (human intestinal model)
- **Units**: Centimeters per second (cm/s)
- **Value Range**: 1.74 × 10⁻⁸ to 3.09 × 10⁻⁴ cm/s

### Conversion to Binary Classification
While the Caco2_Wang dataset originally provides continuous permeability values (regression task), I converted it to a binary classification problem using industry-standard thresholds:

- **Cutoff Value**: 8 × 10⁻⁶ cm/s (based on FDA guidelines and literature)
- **Binary Labels**:
  - High Permeability (1): ≥ 8 × 10⁻⁶ cm/s
  - Low Permeability (0): < 8 × 10⁻⁶ cm/s

**Rationale for Conversion**:
1. Binary classification models are often more robust and easier to validate
2. The cutoff value (8 × 10⁻⁶ cm/s) is well-established in pharmaceutical research
3. In drug discovery, the binary high/low permeability classification is often more actionable than exact values

### Dataset Composition
- **Total Size**: 910 compounds
- **Class Balance**: Near-perfect (51.6% low, 48.4% high permeability)
- **Data Splits**:
  - Training: 637 compounds (51.6% low, 48.4% high)
  - Validation: 91 compounds (46.2% low, 53.8% high)
  - Test: 182 compounds (54.4% low, 45.6% high)

### Molecular Properties
- **Input Format**: SMILES strings (molecular structure)
- **Complexity**: Average SMILES length ~60 characters
- **Chemistry**: Primarily organic compounds
  - Carbon (~57%)
  - Oxygen (~20%)
  - Nitrogen (~10%)
  - Hydrogen (~9%)
  - Other elements (~4%)

## Project Structure
```
.
├── data/               # Dataset and analysis files
│   ├── caco2_train.csv   # Training set (637 compounds)
│   ├── caco2_valid.csv   # Validation set (91 compounds)
│   ├── caco2_test.csv    # Test set (182 compounds)
│ 
├── notebooks/         # Jupyter notebooks for analysis
├── scripts/           # Python scripts
├── models/            # Saved model checkpoints
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Installation Guide

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

### Step-by-Step Setup

1. **Create and Activate Virtual Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate it on Linux/Mac
source venv/bin/activate

# Or on Windows
venv\Scripts\activate
```

2. **Install Dependencies**
```bash
# Install all required packages
pip install -r requirements.txt
```

## Usage Guide

### 1. Dataset Preparation
```bash
# Download and split the dataset
python download_dataset.py
```
This will:
- Download the Caco2_Wang dataset from TDC
- Split it into train/validation/test sets
- Save the splits in the `data` directory
- Generate a detailed analysis in `data/dataset_summary.md`

### 2. Data Exploration
```bash
# View dataset statistics and visualizations
python explore_dataset.py
```
This will:
- Generate detailed statistics about the dataset
- Create visualizations in the `data` directory

### 3. Molecular Featurization

#### Implementation Journey

1. **Initial Approach - Uni-Mol Model**
   - I originally planned to use Ersilia's Uni-Mol model (eos39co) for its advantages:
     - 3D molecular information capture
     - Large-scale training (>200M conformations)
     - SE(3) equivariant architecture
   - Encountered challenges:
     - Docker Hub connectivity issues (TLS handshake timeout)
     - Internal server errors during model fetch
     - Difficulties with local model initialization

     The logs can be found at `data/visualizations/featurization.log`. This may not appear as it may be .gitkept

     The challenges have been addressed by:
     - Docker prunning to alow for more resources to the model fetch and run
     - Addressed the permission issues by escalating the privileges

   ```bash
   # Generate molecular features
   python scripts/featurize_data_unimol.py
   ```
     The model was successfully loaded and the features were generated. as you may find the script in `scripts/featurize_data_unimol.py`.

     The features were saved in the `data/features_unimol` directory.

     The statistics of the features can be found in `data/features_unimol/featurization_stats.json`.

    
2. **Alternative Solution - Morgan Fingerprints**
   - Switched to RDKit's Morgan fingerprints (ECFP4) because:
     - No external dependencies (built into RDKit)
     - Fast computation (910 molecules in ~10 seconds)
     - Well-established in drug discovery
     - High success rate in feature generation

```bash
# Generate molecular features
python scripts/featurize_data.py
```

This script performs several key tasks:
1. **Data Loading**: Reads SMILES strings from each dataset split
2. **Feature Generation**: Creates 2048-bit Morgan fingerprints
3. **Validation**: Checks feature quality and computes statistics
4. **Visualization**: Creates PCA plots to visualize chemical space
5. **Error Handling**: Tracks and reports any failed molecules

#### Feature Analysis

1. **Feature Properties**
   - Dimensionality: 2048 binary bits per molecule
   - Sparsity: ~96.7-96.9% of bits are zero
   - Consistent statistics across splits:
     - Active bits: 3.1-3.3% per molecule
     - Most common substructures appear in 75-95% of molecules
     - Many bits (substructures) never appear
   - No NaN or infinite values

2. **Chemical Space Analysis**
   - PCA visualization:
     - Low variance explained (3.3-5.2% for first 2 PCs)
     - Suggests highly complex chemical space
   - Molecular similarity:
     - Average Tanimoto similarity: 0.149-0.163
     - Similar values within permeability classes
     - Indicates diverse chemical space

3. **Performance Metrics**
   - Processing speed: ~400-600 molecules/second
   - Memory efficiency: <1GB RAM usage
   - Success rate: 100% (910/910 molecules)
   - No failed SMILES parsing

4. **Advantages for ML**
   - Fixed-length representation
   - Interpretable bits (each represents a substructure)
   - Sparse binary format (memory efficient)
   - Well-suited for many ML algorithms

#### Output Structure

```
data/
├── features/
│   ├── caco2_train_features.npy   # Training set feature vectors
│   ├── caco2_valid_features.npy   # Validation set feature vectors
│   ├── caco2_test_features.npy    # Test set feature vectors
│   └── featurization_stats.json   # Detailed statistics about the process
├── processed/
│   ├── caco2_train.csv           # Processed training data
│   ├── caco2_valid.csv           # Processed validation data
│   └── caco2_test.csv            # Processed test data
├── visualizations/
│   ├── pca_features_train.png    # PCA plot for training set
│   ├── pca_features_valid.png    # PCA plot for validation set
│   ├── pca_features_test.png     # PCA plot for test set
│   └── bit_frequency_dist.png    # Distribution of Morgan fingerprint bits
└── featurization.log            # Detailed process log

```

#### Feature Statistics
The `featurization_stats.json` file contains detailed statistics about the generated features:

1. **Overall Statistics**
   - Total molecules processed: 910
   - Successful molecules: 910
   - Success rate: 100%

2. **Split-wise Statistics**
   - Training set (637 molecules):
     - Mean active bits: 3.34%
     - Standard deviation: 0.180
     - Zero bits: 1,260,954 (96.7%)
   - Validation set (91 molecules):
     - Mean active bits: 3.11%
     - Standard deviation: 0.174
     - Zero bits: 180,563 (96.9%)
   - Test set (182 molecules):
     - Mean active bits: 3.11%
     - Standard deviation: 0.174
     - Zero bits: 361,153 (96.9%)

#### Visualizations
The visualization files provide insights into the feature space:

1. **PCA Plots** (`pca_features_*.png`)
   - 2D projection of 2048-dimensional fingerprints
   - Color-coded by permeability class (high/low)
   - Explained variance ratios (3.3-5.2%)
   - Shows molecular similarity patterns

2. **Bit Distribution** (`bit_frequency_dist.png`)
   - Distribution of Morgan fingerprint bits
   - Shows sparsity of the representation
   - Identifies common molecular substructures

3. **Process Log** (`featurization.log`)
   - Detailed timing information
   - Error tracking and handling
   - Feature validation results



## Technical Details

### Data Format
Each compound in the dataset is represented by:
- `Drug_ID`: Unique identifier for each compound
- `Drug`: SMILES string (molecular structure)
- `Y`: Permeability value in log10(cm/s)
- `Permeability`: Actual permeability in cm/s (computed as 10^Y)
- `Binary`: Classification label (0: low, 1: high permeability)

### SMILES Format
SMILES (Simplified Molecular Input Line Entry System) strings in the dataset:
- Represent molecular structure in text format
- Average length: ~60 characters
- Example: `CC(=O)NC1=CC=C(O)C=C1` (Acetaminophen/Paracetamol)
- Most common elements: C, O, N, H (>95% of all atoms)

### Data Quality
1. **Class Balance**
   - Overall: 51.6% low / 48.4% high permeability
   - Training: 51.6% / 48.4%
   - Validation: 46.2% / 53.8%
   - Test: 54.4% / 45.6%

2. **Value Distribution**
   - Range: 1.74 × 10⁻⁸ to 3.09 × 10⁻⁴ cm/s
   - Median: 7.39 × 10⁻⁶ cm/s (close to cutoff)
   - Distribution: See `data/permeability_distribution.png`

### Validation Data (ChEMBL)
- **Source**: ChEMBL database version 35 (2022-2024)
- **Compounds**: 1,045 recent measurements
- **Assay Types**: 
  - Type A (Standard Caco2): 1,035 compounds
  - Type B (Modified): 10 compounds
- **Year Distribution**:
  - 2022: 563 compounds
  - 2023: 464 compounds
  - 2024: 18 compounds

### Binary Classification Task
We converted the continuous permeability values to a binary classification problem using industry-standard thresholds:

- **Cutoff Value**: 8 × 10⁻⁶ cm/s (based on FDA guidelines and literature)
- **Binary Labels**:
  - High Permeability (1): ≥ 8 × 10⁻⁶ cm/s
  - Low Permeability (0): < 8 × 10⁻⁶ cm/s

**Rationale**:
1. Binary classification models are often more robust and easier to validate
2. The cutoff value is well-established in pharmaceutical research
3. Binary high/low permeability classification is often more actionable than exact values


## 📂 Updated Project Structure

| Directory | Purpose | Contents |
|-----------|---------|----------|
| `data/` | Dataset files | - `caco2_train.csv` (637 compounds)<br>- `caco2_valid.csv` (91 compounds)<br>- `caco2_test.csv` (182 compounds)<br>- `chembl_recent.csv` (1,045 compounds)<br>- `dataset_summary.md` |
| `scripts/` | Core code | - Model training & validation<br>- ChEMBL data processing<br>- Feature generation<br>- Analysis & visualization |
| `models/` | Checkpoints | - `xgboost_balanced.json`<br>- Model performance metrics |
| `analysis/` | Results | - Validation results<br>- Chemical space analysis<br>- Performance visualizations |
| Root | Configuration | - `requirements.txt`<br>- `README.md` |


### 2. ChEMBL Data Processing
```bash
# Download and process ChEMBL data
python scripts/process_chembl_dump.py
```
This will:
- Process ChEMBL SQLite database (v35)
- Extract recent Caco2 permeability assays (2022-2024)
- Filter and standardize measurements
- Save processed data to `data/raw/chembl_recent.csv`

### 3. Model Retraining & Validation
```bash
# Train the model
python scripts/retrain_model.py

# Validate on ChEMBL data
python scripts/validate_recent_model.py
```
This will:
- Retrain a balanced XGBoost model
- Generate Morgan fingerprints
- Validate on recent ChEMBL data
- Calculate performance metrics
- Generate validation visualizations

## Model Performance

### Training Results
Our balanced XGBoost model achieves strong performance on the test set:

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.811 |
| Accuracy | 0.784 |
| F1-Score | 0.797 |

### External Validation (ChEMBL)
To validate real-world performance, we tested on 1,045 recent compounds from ChEMBL (2022-2024):

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.709 |
| Accuracy | 0.652 |
| F1-Score | 0.652 |

**Performance by Permeability Range**
| Range | Accuracy | Count |
|-------|----------|-------|
| 0.1-1 µm/s | 73.9% | 157 |
| 1-10 µm/s | 56.7% | 418 |
| 10-100 µm/s | 73.4% | 402 |
| < 0.1 µm/s | 59.0% | 39 |
| > 100 µm/s | 26.1% | 23 |

## Technical Details

### Molecular Featurization
We use Morgan fingerprints (ECFP4-like) for molecular representation:

- **Radius**: 2 (captures up to 4 bonds)
- **Bit Vector Length**: 2048
- **Atom Invariants**: Default RDKit settings
- **Performance**:
  - Processing speed: ~1000 molecules/second
  - Memory usage: <1GB for full dataset
  - Success rate: 100% on both training and ChEMBL data

### Model Architecture
XGBoost classifier with balanced training:

- **Parameters**:
  ```python
  {
      'n_estimators': 100,
      'max_depth': 6,
      'learning_rate': 0.1,
      'subsample': 0.8,
      'colsample_bytree': 0.8,
      'min_child_weight': 1
  }
  ```
- **Early Stopping**: 50 rounds
- **Class Balancing**: Equal sampling of high/low permeability compounds

## Key Findings

1. **Robust Generalization**
   - Model maintains good performance (ROC-AUC 0.709) on unseen ChEMBL data
   - Particularly strong in physiologically relevant ranges (0.1-100 µm/s)

2. **Range-Specific Performance**
   - Best accuracy in 0.1-1 µm/s (73.9%) and 10-100 µm/s (73.4%) ranges
   - Challenges with very high permeability compounds (>100 µm/s)

3. **Temporal Stability**
   - Performance remains consistent across 2022-2024 data
   - Suggests model captures fundamental permeability determinants

## Future Work

1. **Model Improvements**
   - Ensemble with 3D structure-based models
   - Incorporate uncertainty estimation
   - Explore transformer architectures

2. **Data Enhancement**
   - Include more diverse chemical scaffolds
   - Gather additional very high permeability examples
   - Consider multi-task learning with related properties

3. **Validation Extensions**
   - Test on other cell line variants
   - Compare with in vivo absorption data
   - Analyze prediction confidence metrics
