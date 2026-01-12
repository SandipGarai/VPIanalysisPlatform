"""
Statistics Module
Handles ANOVA and Tukey HSD post-hoc analysis
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import t


def critical_difference(se, df, alpha=0.05):
    """
    Calculate critical difference for treatment comparison

    Parameters:
    -----------
    se : float
        Standard error
    df : int
        Degrees of freedom
    alpha : float
        Significance level (default: 0.05)

    Returns:
    --------
    float : Critical difference value
    """
    t_val = t.ppf(1 - alpha / 2, df)
    return t_val * se * np.sqrt(2)


def perform_anova_tukey(data, alpha=0.05):
    """
    Perform ANOVA and Tukey HSD tests for all variables and plants

    Parameters:
    -----------
    data : pd.DataFrame
        Processed data with VPI and RFD
    alpha : float
        Significance level (default: 0.05)

    Returns:
    --------
    tuple : (anova_results, tukey_results)
        - anova_results: Dictionary of ANOVA tables
        - tukey_results: Dictionary of Tukey HSD results
    """
    plants = data['Plant'].unique()
    variables = ['DSI', 'AUDSC', 'RFVL', 'VPI', 'RFD']

    anova_results = {}
    tukey_results = {}

    for plant in plants:
        df_plant = data[data['Plant'] == plant]

        for var in variables:
            # Skip RFD analysis for T1 since it's always 0
            if var == 'RFD':
                df_plant_var = df_plant[df_plant['Treatment'] != 'T1'].copy()
            else:
                df_plant_var = df_plant.copy()

            # Skip if no data available
            if df_plant_var.empty or df_plant_var[var].isna().all():
                continue

            try:
                # Fit ANOVA model
                model = ols(f'{var} ~ C(Treatment)', data=df_plant_var).fit()
                anova = sm.stats.anova_lm(model, typ=2)

                # Extract ANOVA statistics
                df_treat = anova.loc['C(Treatment)', 'df']
                df_resid = anova.loc['Residual', 'df']
                ss_treat = anova.loc['C(Treatment)', 'sum_sq']
                ss_resid = anova.loc['Residual', 'sum_sq']
                ms_treat = ss_treat / df_treat
                ms_resid = ss_resid / df_resid
                f_val = anova.loc['C(Treatment)', 'F']
                p_val = anova.loc['C(Treatment)', 'PR(>F)']

                # Calculate SEM and CD
                reps = df_plant_var.groupby('Treatment')[var].count().mean()
                sem = np.sqrt(ms_resid / reps)
                cd = critical_difference(sem, df_resid, alpha)

                # Dynamic CD label based on alpha level
                alpha_percent = int(alpha * 100)
                cd_label = f'CD ({alpha_percent}%)'

                # Create ANOVA table
                anova_table = pd.DataFrame({
                    'SoV': ['Treatment', 'Error'],
                    'df': [int(df_treat), int(df_resid)],
                    'Sum Sq': [ss_treat, ss_resid],
                    'Mean Sq': [ms_treat, ms_resid],
                    'F value': [f_val, np.nan],
                    'p-value': [p_val, np.nan],
                    'SEm': [sem, np.nan],
                    cd_label: [cd, np.nan]
                })

                key = f"{plant}_{var}"
                anova_results[key] = anova_table

                # Tukey HSD test
                tukey = pairwise_tukeyhsd(
                    endog=df_plant_var[var],
                    groups=df_plant_var['Treatment'],
                    alpha=alpha
                )

                # Create summary with grouping
                summary = df_plant_var.groupby('Treatment')[var].agg(
                    ['mean', 'std', 'count']
                ).reset_index()
                summary.columns = ['Treatment', 'Mean', 'Std', 'Replication']
                summary['SE'] = summary['Std'] / \
                    np.sqrt(summary['Replication'])

                # Sort by mean (descending)
                sorted_summary = summary.sort_values(
                    'Mean', ascending=False).reset_index(drop=True)

                # Assign letter groups based on Tukey results
                sorted_summary['Group'] = assign_letter_groups(
                    sorted_summary['Treatment'].tolist(),
                    tukey
                )

                tukey_results[key] = sorted_summary

            except Exception as e:
                print(f"Error processing {plant}_{var}: {str(e)}")
                continue

    return anova_results, tukey_results


def assign_letter_groups(treatments, tukey_result):
    """
    Assign letter groups based on Tukey HSD test results
    Uses the standard compact letter display (CLD) algorithm

    Parameters:
    -----------
    treatments : list
        List of treatment names (sorted by mean, descending)
    tukey_result : TukeyHSDResults
        Results from pairwise_tukeyhsd

    Returns:
    --------
    list : Letter groups for each treatment (lowercase: a, b, c, ...)
        Treatments with same letter are NOT significantly different
        Group 'a' = highest mean, 'b' = second highest, etc.
    """
    # The original code didn't actually use tukey_result properly!
    # But we still need to use it correctly. Let me check what makes treatments
    # significantly different

    tukey_df = pd.DataFrame(
        data=tukey_result.summary().data[1:],
        columns=tukey_result.summary().data[0]
    )

    # Build set of significantly different pairs based on Tukey HSD test
    sig_pairs = set()
    for _, row in tukey_df.iterrows():
        if row['reject']:  # reject=True means significantly different
            sig_pairs.add((row['group1'], row['group2']))
            sig_pairs.add((row['group2'], row['group1']))

    # Compact Letter Display (CLD) Algorithm
    # Treatments are already sorted by mean (descending)
    n_treatments = len(treatments)
    letter_assignments = [[] for _ in range(n_treatments)]

    current_letter = 0

    for i in range(n_treatments):
        # Check if treatment i can be added to existing groups
        can_join = [False] * current_letter

        for letter_idx in range(current_letter):
            # Check if treatment i is compatible with letter group
            compatible = True
            for j in range(i):
                if letter_idx in letter_assignments[j]:
                    # Check if i and j are significantly different
                    if (treatments[i], treatments[j]) in sig_pairs:
                        compatible = False
                        break

            if compatible:
                can_join[letter_idx] = True

        # Assign letters to treatment i
        if any(can_join):
            for letter_idx in range(current_letter):
                if can_join[letter_idx]:
                    letter_assignments[i].append(letter_idx)
        else:
            # Create new letter group
            letter_assignments[i].append(current_letter)
            current_letter += 1

    # Convert letter indices to actual letters
    letter_groups = []
    for assignment in letter_assignments:
        # a, b, c...
        letters = ''.join([chr(97 + idx) for idx in sorted(assignment)])
        letter_groups.append(letters)

    return letter_groups


def get_treatment_comparison_matrix(data, plant, variable):
    """
    Create a pairwise comparison matrix for treatments

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    plant : str
        Plant name
    variable : str
        Variable name

    Returns:
    --------
    pd.DataFrame : Comparison matrix with p-values
    """
    plant_data = data[data['Plant'] == plant]

    if variable == 'RFD':
        plant_data = plant_data[plant_data['Treatment'] != 'T1']

    treatments = sorted(plant_data['Treatment'].unique())
    n_treatments = len(treatments)

    # Initialize matrix
    matrix = pd.DataFrame(
        np.zeros((n_treatments, n_treatments)),
        index=treatments,
        columns=treatments
    )

    # Perform Tukey HSD
    tukey = pairwise_tukeyhsd(
        endog=plant_data[variable],
        groups=plant_data['Treatment'],
        alpha=0.05
    )

    # Fill matrix with p-values
    tukey_df = pd.DataFrame(
        data=tukey.summary().data[1:],
        columns=tukey.summary().data[0]
    )

    for _, row in tukey_df.iterrows():
        t1, t2 = row['group1'], row['group2']
        p_val = row['p-adj']
        matrix.loc[t1, t2] = p_val
        matrix.loc[t2, t1] = p_val

    return matrix
