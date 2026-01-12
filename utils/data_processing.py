"""
Data Processing Module
Handles RFD calculation and VPI computation using PCA
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_rfd(data):
    """
    Calculate Relative Fold Reduction (RFD) for each treatment compared to T1 (control)

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe with columns: Plant, Treatment, RFVL

    Returns:
    --------
    pd.DataFrame : Data with RFD column added
    """
    data_with_rfd = data.copy()
    data_with_rfd['RFD'] = np.nan

    plants = data['Plant'].unique()

    for plant in plants:
        plant_data = data_with_rfd[data_with_rfd['Plant'] == plant].copy()

        # Get T1 values for this plant
        t1_data = plant_data[plant_data['Treatment'] == 'T1'].copy()

        for idx, row in plant_data.iterrows():
            if row['Treatment'] == 'T1':
                # T1 gets 0 value (baseline)
                data_with_rfd.loc[idx, 'RFD'] = 0
            else:
                # Find corresponding T1 value (by replication/row position)
                treatment_group = plant_data[plant_data['Treatment']
                                             == row['Treatment']]
                treatment_position = treatment_group.index.get_loc(idx)

                if treatment_position < len(t1_data):
                    t1_rfvl = t1_data.iloc[treatment_position]['RFVL']
                    current_rfvl = row['RFVL']

                    if pd.notna(t1_rfvl) and pd.notna(current_rfvl) and current_rfvl != 0:
                        rfd_value = t1_rfvl / current_rfvl
                        data_with_rfd.loc[idx, 'RFD'] = rfd_value

    return data_with_rfd


def compute_vpi_pca(data, dpi=300):
    """
    Compute VPI using Principal Component Analysis

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe with DSI and RFVL columns
    dpi : int
        DPI for plots (default: 300)

    Returns:
    --------
    tuple : (processed_data, pca_results)
        - processed_data: DataFrame with VPI and RFD columns
        - pca_results: Dictionary containing PCA analysis results for each plant
    """
    # Features for PCA
    features = ['DSI', 'RFVL']
    plants = data['Plant'].unique()

    # Initialize processed data
    data_processed = data.copy()
    data_processed['VPI'] = np.nan

    # Store PCA results
    pca_results = {}

    # Set global font properties
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'

    # Binomial nomenclature plants (to be italicized)
    binomial_plants = ["N. benthamiana", "A. thaliana", "S. lycopersicum"]

    for plant in plants:
        # Extract plant data
        data_plant = data[data['Plant'] == plant].copy()
        data_clean = data_plant.dropna(subset=features).copy()

        if len(data_clean) < 2:
            continue

        # Standardize features
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(data_clean[features])

        # Apply PCA
        pca = PCA()
        principal_components = pca.fit_transform(x_scaled)
        pca_df = pd.DataFrame(
            principal_components,
            columns=[f'PC{i+1}' for i in range(principal_components.shape[1])],
            index=data_clean.index
        )

        # Compute VPI using inverse of PC1 (normalized)
        data_clean['VPI'] = 1 - \
            MinMaxScaler().fit_transform(pca_df[['PC1']])
        data_processed.loc[data_clean.index, 'VPI'] = data_clean['VPI']

        # Create Scree Plot
        scree_fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(
            range(1, len(pca.explained_variance_ratio_) + 1),
            pca.explained_variance_ratio_ * 100,
            marker='o',
            color='black',
            linewidth=2
        )
        ax.set_xlabel('Principal Component', fontsize=22,
                      fontweight='bold', fontname='Times New Roman')
        ax.set_ylabel('Explained Variance (%)', fontsize=22,
                      fontweight='bold', fontname='Times New Roman')
        ax.set_xticks(np.arange(1, len(pca.explained_variance_ratio_) + 1, 1))
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.setp(ax.get_xticklabels(), fontweight='bold',
                 fontname='Times New Roman')
        plt.setp(ax.get_yticklabels(), fontweight='bold',
                 fontname='Times New Roman')
        ax.grid(True, linewidth=1.2)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        plt.tight_layout()

        # Create PC1 Loadings Plot
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(len(features))],
            index=features
        )

        loadings_fig, ax2 = plt.subplots(figsize=(8, 6))
        sns.barplot(
            x=loadings.index, y=loadings['PC1'], hue=loadings.index,
            palette='pastel', edgecolor='black', ax=ax2, legend=False)
        ax2.set_xlabel('Variable', fontsize=22,
                       fontweight='bold', fontname='Times New Roman')
        ax2.set_ylabel('Loading Coefficient', fontsize=22,
                       fontweight='bold', fontname='Times New Roman')
        # Set x-axis ticks and labels properly
        ax2.set_xticks(range(len(loadings.index)))
        ax2.set_xticklabels(loadings.index, fontsize=20,
                            fontweight='bold', fontname='Times New Roman')
        # Set y-axis ticks and labels properly
        yticks = ax2.get_yticks()
        ax2.set_yticks(yticks)
        ax2.set_yticklabels(
            [f"{y:.2f}" for y in yticks],
            fontsize=20,
            fontweight='bold',
            fontname='Times New Roman'
        )
        for spine in ax2.spines.values():
            spine.set_linewidth(1.5)
        plt.tight_layout()

        # Store results
        pca_results[plant] = {
            'explained_variance': pca.explained_variance_ratio_,
            'loadings': loadings,
            'scree_plot': scree_fig,
            'loadings_plot': loadings_fig,
            'pca_object': pca
        }

        plt.close('all')  # Clean up

    # Calculate RFD
    data_processed = calculate_rfd(data_processed)

    return data_processed, pca_results


def validate_data(data):
    """
    Validate input data structure and content

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe to validate

    Returns:
    --------
    tuple : (is_valid, error_messages)
    """
    errors = []

    # Check required columns
    required_cols = ['Plant', 'Treatment', 'DSI', 'AUDSC', 'RFVL']
    missing_cols = [col for col in required_cols if col not in data.columns]

    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")

    # Check for null values in critical columns
    if data[['Plant', 'Treatment']].isnull().any().any():
        errors.append("Null values found in Plant or Treatment columns")

    # Check data types
    try:
        data[['DSI', 'AUDSC', 'RFVL']] = data[[
            'DSI', 'AUDSC', 'RFVL']].astype(float)
    except Exception as e:
        errors.append(f"Non-numeric values in measurement columns: {str(e)}")

    # Check value ranges
    if (data['DSI'] < 0).any() or (data['DSI'] > 100).any():
        errors.append("DSI values should be between 0 and 100")

    if (data['RFVL'] < 0).any():
        errors.append("RFVL values should be non-negative")

    is_valid = len(errors) == 0

    return is_valid, errors
