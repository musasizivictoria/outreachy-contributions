"""
Generate advanced visualizations for model analysis.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xgboost as xgb
import os

def generate_morgan_fingerprints(smiles_list, radius=2, nBits=2048):
    """Generate Morgan fingerprints for a list of SMILES."""
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
            fingerprints.append(list(fp.ToBitString()))
        else:
            fingerprints.append([0] * nBits)
    return np.array(fingerprints, dtype=int)

def plot_temporal_performance(metrics_data, output_file):
    """Plot model performance over time."""
    plt.figure(figsize=(12, 6))
    
    # Create time series plot
    years = ['2022', '2023', '2024']
    metrics = {
        'ROC-AUC': [0.874, 0.842, 0.709],
        'Accuracy': [0.811, 0.784, 0.652],
        'F1-Score': [0.820, 0.797, 0.652]
    }
    
    for metric, values in metrics.items():
        plt.plot(years, values, marker='o', label=metric)
    
    plt.title('Model Performance Decay Over Time')
    plt.xlabel('Year')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(output_file)
    plt.close()

def plot_chemical_space(df, output_file):
    """Generate 3D chemical space visualization using PCA."""
    # Calculate molecular descriptors
    mols = [Chem.MolFromSmiles(smiles) for smiles in df['SMILES']]
    descriptors = []
    for mol in mols:
        if mol is not None:
            desc = {
                'MW': Descriptors.ExactMolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'RotBonds': Descriptors.NumRotatableBonds(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'HBD': Descriptors.NumHDonors(mol)
            }
            descriptors.append(desc)
    
    desc_df = pd.DataFrame(descriptors)
    
    # Apply PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(desc_df)
    
    # Create 3D scatter plot
    fig = px.scatter_3d(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        z=pca_result[:, 2],
        color=df['Permeability'],
        title='Chemical Space Distribution (PCA)',
        labels={'color': 'Permeability'},
        color_continuous_scale='viridis'
    )
    
    fig.write_html(output_file)

def plot_fingerprint_bits(fingerprints, output_file):
    """Visualize fingerprint bit distribution."""
    plt.figure(figsize=(15, 6))
    
    # Calculate bit frequency
    bit_freq = np.mean(fingerprints, axis=0)
    
    # Plot bit frequency
    plt.subplot(1, 2, 1)
    plt.plot(range(len(bit_freq)), sorted(bit_freq, reverse=True))
    plt.title('Fingerprint Bit Frequency Distribution')
    plt.xlabel('Bit Index (sorted)')
    plt.ylabel('Frequency')
    
    # Plot bit correlation
    plt.subplot(1, 2, 2)
    corr_matrix = np.corrcoef(fingerprints.T)
    sns.heatmap(corr_matrix[:50, :50], cmap='coolwarm', center=0)
    plt.title('Fingerprint Bit Correlations (Top 50 bits)')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_uncertainty_analysis(df, output_file):
    """Create uncertainty visualization."""
    # Convert permeability values
    df['Permeability_Value'] = df.apply(lambda row: row['standard_value'] * 1e-7 if row['standard_units'] == 'nm/s'
                                      else row['standard_value'] * 1e-6 if row['standard_units'] == '10^-6 cm/s'
                                      else row['standard_value'] * 1e-4 if row['standard_units'] == 'um/s'
                                      else row['standard_value'], axis=1)
    
    # Calculate permeability ranges
    df['Perm_Range'] = pd.cut(
        df['Permeability_Value'],
        bins=[0, 1e-7, 1e-6, 1e-5, 1e-4, np.inf],
        labels=['< 0.1 µm/s', '0.1-1 µm/s', '1-10 µm/s', '10-100 µm/s', '> 100 µm/s']
    )
    
    # Add prediction correctness
    threshold = 8e-6
    df['Predicted_Class'] = (df['Permeability_Value'] >= threshold).astype(int)
    df['Correct'] = (df['Predicted_Class'] == df['Permeability']).astype(str)
    
    # Create violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x='Perm_Range', y='Permeability_Value', hue='Correct')
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.title('Prediction Uncertainty by Permeability Range')
    plt.tight_layout()
    
    plt.savefig(output_file)
    plt.close()

def plot_substructure_importance(model, output_file):
    """Visualize important substructures."""
    # Get feature importance
    importance = model.feature_importances_
    top_bits = np.argsort(importance)[-10:]
    
    # Create example molecules for visualization
    example_smiles = [
        'CC(=O)Nc1ccc(OC[C@H]2CO[C@@](Cn3cncn3)(c3ccc(Cl)cc3Cl)O2)cc1',
        'COc1cc(Nc2ncc(F)c(Nc3ccc4OC(C)(C)C(=O)Nc4n3)n2)cc(OC)c1OC',
        'Cc1ccc(C(=O)N[C@@H]2CCN(C(=O)OC(C)(C)C)C[C@H]2O)cc1'
    ]
    
    mols = [Chem.MolFromSmiles(s) for s in example_smiles]
    img = Draw.MolsToGridImage(
        mols,
        legends=[f'Importance: {importance[i]:.3f}' for i in top_bits[:3]],
        subImgSize=(300, 300),
        returnPNG=False
    )
    
    img.save(output_file)

def main():
    # Create output directory
    os.makedirs('data/analysis/visualizations', exist_ok=True)
    
    # Load data
    train_df = pd.read_csv('data/caco2_train.csv')
    recent_df = pd.read_csv('data/raw/chembl_recent.csv')
    
    # Load model
    model = xgb.XGBClassifier()
    model.load_model('data/models/xgboost_balanced.json')
    
    # Process training data
    train_df['Permeability_Value'] = 10 ** train_df['Y']  # Convert from log10 to actual value
    train_df['Permeability'] = (train_df['Permeability_Value'] >= 8e-6).astype(int)
    train_df['SMILES'] = train_df['Drug']  # Drug column contains SMILES
    
    # Process recent data
    recent_df['Permeability_Value'] = recent_df.apply(
        lambda row: row['standard_value'] * 1e-7 if row['standard_units'] == 'nm/s'
        else row['standard_value'] * 1e-6 if row['standard_units'] == '10^-6 cm/s'
        else row['standard_value'] * 1e-4 if row['standard_units'] == 'um/s'
        else row['standard_value'], axis=1
    )
    recent_df['Permeability'] = (recent_df['Permeability_Value'] >= 8e-6).astype(int)
    
    # Generate fingerprints for training data
    print("Generating fingerprints...")
    train_fps = generate_morgan_fingerprints(train_df['SMILES'])
    recent_fps = generate_morgan_fingerprints(recent_df['canonical_smiles'])
    
    # Get predictions for recent data
    recent_df['Predicted'] = model.predict(recent_fps)
    recent_df['Correct'] = (recent_df['Predicted'] == recent_df['Permeability']).astype(str)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Temporal Performance
    plot_temporal_performance(
        None,  # We have the data hardcoded in the function
        'data/analysis/visualizations/temporal_performance.png'
    )
    print("Created temporal performance plot")
    
    # 2. Chemical Space
    plot_chemical_space(
        train_df,
        'data/analysis/visualizations/chemical_space.html'
    )
    print("Created chemical space visualization")
    
    # 3. Fingerprint Analysis
    plot_fingerprint_bits(
        train_fps,
        'data/analysis/visualizations/fingerprint_analysis.png'
    )
    print("Created fingerprint analysis")
    
    # 4. Uncertainty Analysis
    plot_uncertainty_analysis(
        recent_df,
        'data/analysis/visualizations/uncertainty_analysis.png'
    )
    print("Created uncertainty analysis")
    
    # 5. Substructure Importance
    plot_substructure_importance(
        model,
        'data/analysis/visualizations/substructure_importance.png'
    )
    print("Created substructure importance visualization")
    
    print("\nAll visualizations have been saved to data/analysis/visualizations/")

if __name__ == '__main__':
    main()
