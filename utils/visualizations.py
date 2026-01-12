"""
Visualizations Module
Publication-quality plots for VPI analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_scree_plot(pca, plant_name, dpi=300):
    """
    Create scree plot for PCA

    Parameters:
    -----------
    pca : sklearn.decomposition.PCA
        Fitted PCA object
    plant_name : str
        Name of the plant
    dpi : int
        Plot resolution

    Returns:
    --------
    matplotlib.figure.Figure : Scree plot
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

    ax.plot(
        range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_ * 100,
        marker='o',
        color='black',
        linewidth=2
    )

    ax.set_xlabel('Principal Component', fontsize=22, fontweight='bold')
    ax.set_ylabel('Explained Variance (%)', fontsize=22, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, linewidth=1.2)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()

    return fig


def plot_pca_loadings(loadings, plant_name, dpi=300):
    """
    Create PC1 loadings bar plot

    Parameters:
    -----------
    loadings : pd.DataFrame
        PCA loadings dataframe
    plant_name : str
        Name of the plant
    dpi : int
        Plot resolution

    Returns:
    --------
    matplotlib.figure.Figure : Loadings plot
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

    sns.barplot(
        x=loadings.index,
        y=loadings['PC1'],
        hue=loadings.index,
        palette='pastel',
        edgecolor='black',
        ax=ax,
        legend=False
    )

    ax.set_xlabel('Variable', fontsize=22, fontweight='bold')
    ax.set_ylabel('Loading Coefficient', fontsize=22, fontweight='bold')
    ax.set_xticks(range(len(loadings.index)))
    ax.set_xticklabels(loadings.index, fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=20)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()

    return fig


def plot_comparative_charts(data, tukey_results=None, dpi=300):
    """
    Create comparative bar charts for DSI, AUDSC, and RFVL with letter groupings

    Parameters:
    -----------
    data : pd.DataFrame
        Processed data with treatment effects
    tukey_results : dict, optional
        Dictionary of Tukey HSD results with letter groups
    dpi : int
        Plot resolution

    Returns:
    --------
    matplotlib.figure.Figure : Comparative charts
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'

    variables = ['DSI', 'AUDSC', 'RFVL']
    plants = sorted(data['Plant'].unique())
    treatment_order = sorted(data['Treatment'].unique())

    binomial_plants = ["N. benthamiana", "A. thaliana", "S. lycopersicum"]

    y_axis_labels = {
        "AUDSC": "Percent-Days",
        "DSI": "Percent",
        "RFVL": "Viral Transcript"
    }

    rows = len(plants)
    cols = len(variables)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), dpi=dpi)

    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for i, plant in enumerate(plants):
        plant_data = data[data['Plant'] == plant]

        for j, variable in enumerate(variables):
            ax = axes[i][j]

            # Calculate means and SE for each treatment
            summary = plant_data.groupby('Treatment')[variable].agg([
                'mean', 'sem']).reset_index()
            summary.columns = ['Treatment', 'Mean', 'SE']
            summary['Treatment'] = pd.Categorical(
                summary['Treatment'],
                categories=treatment_order,
                ordered=True
            )
            summary = summary.sort_values('Treatment')

            # Get letter groups if available
            letter_groups = {}
            if tukey_results:
                key = f"{plant}_{variable}"
                if key in tukey_results:
                    tukey_df = tukey_results[key]
                    for _, row in tukey_df.iterrows():
                        letter_groups[row['Treatment']] = row['Group']

            # Create bar plot
            sns.barplot(
                x='Treatment',
                y='Mean',
                hue='Treatment',
                data=summary,
                ax=ax,
                palette='Set2',
                legend=False
            )

            # Add error bars and letter groups
            y_max = summary['Mean'].max() + summary['SE'].max() * 1.2
            for idx, row in summary.iterrows():
                # Error bars
                ax.errorbar(
                    idx,
                    row['Mean'],
                    yerr=row['SE'],
                    fmt='none',
                    ecolor='black',
                    capsize=5
                )

                # Add letter group
                if row['Treatment'] in letter_groups:
                    text_y = row['Mean'] + row['SE'] + (y_max * 0.02)
                    ax.text(
                        idx, text_y,
                        str(letter_groups[row['Treatment']]
                            ).lower(),  # Lowercase display
                        ha='center',
                        va='bottom',
                        fontsize=25,
                        fontweight='bold',
                        color='black',
                        fontname='Times New Roman'
                    )

            # Formatting
            ax.set_ylim(0, y_max + (y_max * 0.1))

            if i < rows - 1:
                ax.set_xlabel("")
            else:
                ax.set_xlabel("Treatment", fontsize=25, fontweight='bold')

            ax.set_ylabel(y_axis_labels.get(variable, ""),
                          fontsize=25, fontweight='bold')
            # Set x-axis ticks before labels
            ax.set_xticks(range(len(treatment_order)))
            ax.set_xticklabels(treatment_order, rotation=45,
                               ha='right', fontsize=25)
            ax.tick_params(axis='both', which='major', labelsize=25)

            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

    # Add row and column labels
    for i, plant in enumerate(plants):
        style = 'italic' if plant in binomial_plants else 'normal'
        fig.text(
            0.04, 1 - (i + 0.5) / rows,
            plant,
            ha='center',
            va='center',
            fontsize=30,
            fontweight='bold',
            rotation=90,
            style=style
        )

    for j, variable in enumerate(variables):
        fig.text(
            (j + 0.5) / cols, 1.01,
            variable,
            ha='center',
            va='center',
            fontsize=25,
            fontweight='bold'
        )

    plt.tight_layout(rect=[0.06, 0.03, 1, 0.95])

    return fig


def plot_vpi_chart(data, tukey_results=None, dpi=300):
    """
    Create VPI comparison chart across all plants with letter groupings

    Parameters:
    -----------
    data : pd.DataFrame
        Processed data with VPI values
    tukey_results : dict, optional
        Dictionary of Tukey HSD results with letter groups
    dpi : int
        Plot resolution

    Returns:
    --------
    matplotlib.figure.Figure : VPI chart
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'

    binomial_plants = ["N. benthamiana", "A. thaliana", "S. lycopersicum"]
    treatment_order = sorted(data['Treatment'].unique())

    # Calculate summary statistics
    summary = data.groupby(['Plant', 'Treatment'])[
        'VPI'].agg(['mean', 'sem']).reset_index()
    summary.columns = ['Plant', 'Treatment', 'Mean', 'SE']
    summary['Treatment'] = pd.Categorical(
        summary['Treatment'],
        categories=treatment_order,
        ordered=True
    )
    summary = summary.sort_values(['Treatment', 'Plant'])

    fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi)

    # Create grouped bar plot
    sns.barplot(
        x='Treatment',
        y='Mean',
        hue='Plant',
        data=summary,
        palette='Set2',
        ax=ax,
        errwidth=0
    )

    # Calculate y_max for proper spacing
    y_max = summary['Mean'].max() + summary['SE'].max() * 1.2
    ax.set_ylim(0, y_max + (y_max * 0.1))

    # Get letter groups for each plant
    letter_groups = {}
    if tukey_results:
        plants = sorted(data['Plant'].unique())
        for plant in plants:
            key = f"{plant}_VPI"
            if key in tukey_results:
                tukey_df = tukey_results[key]
                for _, row in tukey_df.iterrows():
                    letter_groups[(plant, row['Treatment'])] = row['Group']

    # Add error bars and letter groups manually
    for i, (treatment, group_df) in enumerate(summary.groupby('Treatment')):
        group_df = group_df.reset_index(drop=True)
        n = len(group_df)
        offsets = np.linspace(-0.2, 0.2, n)

        for j, (_, row) in enumerate(group_df.iterrows()):
            bar_x = i + offsets[j]
            bar_y = row['Mean']
            bar_se = row['SE']

            # Error bars
            ax.errorbar(
                bar_x, bar_y,
                yerr=bar_se,
                fmt='none',
                ecolor='black',
                capsize=5,
                linewidth=1.5
            )

            # Letter groups
            if (row['Plant'], row['Treatment']) in letter_groups:
                text_y = bar_y + bar_se + (y_max * 0.02)
                ax.text(
                    bar_x, text_y,
                    # Lowercase display
                    str(letter_groups[(row['Plant'],
                        row['Treatment'])]).lower(),
                    ha='center',
                    va='bottom',
                    fontsize=25,
                    fontweight='bold',
                    fontname='Times New Roman',
                    color='black'
                )

    # Formatting
    ax.set_xlabel("Treatment", fontsize=25, fontweight='bold')
    ax.set_ylabel("Protection Index Value", fontsize=25, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.set_xticks(range(len(treatment_order)))
    ax.set_xticklabels(treatment_order, rotation=45, ha='right', fontsize=25)

    # Format legend
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles, labels,
        title="Plant",
        title_fontsize=25,
        prop={'family': 'Times New Roman', 'size': 20, 'weight': 'bold'},
        loc='upper left'
    )

    # Italicize binomial names
    for i, label in enumerate(labels):
        if label in binomial_plants:
            legend.get_texts()[i].set_style('italic')

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()

    return fig


def plot_rfd_chart(data, tukey_results=None, dpi=300):
    """
    Create RFD comparison chart across all plants (excluding T1) with letter groupings

    Parameters:
    -----------
    data : pd.DataFrame
        Processed data with RFD values
    tukey_results : dict, optional
        Dictionary of Tukey HSD results with letter groups
    dpi : int
        Plot resolution

    Returns:
    --------
    matplotlib.figure.Figure : RFD chart
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'

    binomial_plants = ["N. benthamiana", "A. thaliana", "S. lycopersicum"]

    # Exclude T1 (control)
    data_rfd = data[data['Treatment'] != 'T1'].copy()
    treatment_order = sorted(data_rfd['Treatment'].unique())

    # Calculate summary statistics
    summary = data_rfd.groupby(['Plant', 'Treatment'])[
        'RFD'].agg(['mean', 'sem']).reset_index()
    summary.columns = ['Plant', 'Treatment', 'Mean', 'SE']
    summary['Treatment'] = pd.Categorical(
        summary['Treatment'],
        categories=treatment_order,
        ordered=True
    )
    summary = summary.sort_values(['Treatment', 'Plant'])

    fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi)

    # Create grouped bar plot
    sns.barplot(
        x='Treatment',
        y='Mean',
        hue='Plant',
        data=summary,
        palette='Set2',
        ax=ax,
        errwidth=0
    )

    # Calculate y_max for proper spacing
    y_max = summary['Mean'].max() + summary['SE'].max() * 1.2
    ax.set_ylim(0, y_max + (y_max * 0.1))

    # Get letter groups for each plant
    letter_groups = {}
    if tukey_results:
        plants = sorted(data['Plant'].unique())
        for plant in plants:
            key = f"{plant}_RFD"
            if key in tukey_results:
                tukey_df = tukey_results[key]
                for _, row in tukey_df.iterrows():
                    letter_groups[(plant, row['Treatment'])] = row['Group']

    # Add error bars and letter groups manually
    for i, (treatment, group_df) in enumerate(summary.groupby('Treatment')):
        group_df = group_df.reset_index(drop=True)
        n = len(group_df)
        offsets = np.linspace(-0.2, 0.2, n)

        for j, (_, row) in enumerate(group_df.iterrows()):
            bar_x = i + offsets[j]
            bar_y = row['Mean']
            bar_se = row['SE']

            # Error bars
            ax.errorbar(
                bar_x, bar_y,
                yerr=bar_se,
                fmt='none',
                ecolor='black',
                capsize=5,
                linewidth=1.5
            )

            # Letter groups
            if (row['Plant'], row['Treatment']) in letter_groups:
                text_y = bar_y + bar_se + (y_max * 0.02)
                ax.text(
                    bar_x, text_y,
                    # Lowercase display
                    str(letter_groups[(row['Plant'],
                        row['Treatment'])]).lower(),
                    ha='center',
                    va='bottom',
                    fontsize=25,
                    fontweight='bold',
                    fontname='Times New Roman',
                    color='black'
                )

    # Formatting
    ax.set_xlabel("Treatment", fontsize=25, fontweight='bold')
    ax.set_ylabel("Fold Reduction", fontsize=25, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.set_xticks(range(len(treatment_order)))
    ax.set_xticklabels(treatment_order, rotation=45, ha='right', fontsize=25)

    # Format legend
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles, labels,
        title="Plant",
        title_fontsize=25,
        prop={'family': 'Times New Roman', 'size': 20, 'weight': 'bold'},
        loc='upper left'
    )

    # Italicize binomial names
    for i, label in enumerate(labels):
        if label in binomial_plants:
            legend.get_texts()[i].set_style('italic')

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()

    return fig


def generate_html_report(data, processed_data, pca_results, anova_results, tukey_results,
                         plot_dpi=300):
    """
    Generate comprehensive HTML report with all tables and figures

    Parameters:
    -----------
    data : pd.DataFrame
        Original data
    processed_data : pd.DataFrame
        Processed data with VPI and RFD
    pca_results : dict
        PCA analysis results
    anova_results : dict
        ANOVA results
    tukey_results : dict
        Tukey HSD results
    plot_dpi : int
        Plot resolution

    Returns:
    --------
    str : HTML content
    """
    from io import BytesIO
    import base64

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>VPI Analysis Report</title>
        <style>
            body {
                font-family: 'Times New Roman', serif;
                margin: 40px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2E7D32;
                text-align: center;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }
            h2 {
                color: #1B5E20;
                margin-top: 30px;
                border-left: 5px solid #4CAF50;
                padding-left: 10px;
            }
            h3 {
                color: #388E3C;
                margin-top: 20px;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }
            th {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            .figure {
                margin: 30px 0;
                text-align: center;
            }
            .figure img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                padding: 5px;
            }
            .figure-caption {
                font-style: italic;
                margin-top: 10px;
                color: #555;
            }
            .metric-box {
                background-color: #E8F5E9;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #4CAF50;
                margin: 15px 0;
            }
            .date {
                text-align: right;
                color: #666;
                font-size: 0.9em;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌿 VPI Analysis Report</h1>
            <p class="date">Generated on: """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            
            <h2>1. Study Overview</h2>
            <div class="metric-box">
                <p><strong>Total Observations:</strong> """ + str(len(data)) + """</p>
                <p><strong>Plant Species:</strong> """ + ", ".join(data['Plant'].unique()) + """</p>
                <p><strong>Treatments:</strong> """ + ", ".join(sorted(data['Treatment'].unique())) + """</p>
                <p><strong>Variables Analyzed:</strong> DSI, AUDSC, RFVL, VPI, RFD</p>
            </div>
    """

    # Add PCA results
    html_content += "\n<h2>2. Principal Component Analysis (PCA)</h2>\n"

    for plant, results in pca_results.items():
        html_content += f"\n<h3>{plant}</h3>\n"

        # Explained variance
        exp_var = results['explained_variance']
        html_content += f"""
        <div class="metric-box">
            <p><strong>PC1 Explained Variance:</strong> {exp_var[0]*100:.2f}%</p>
            <p><strong>PC2 Explained Variance:</strong> {exp_var[1]*100:.2f}%</p>
        </div>
        """

        # Scree plot
        fig = results['scree_plot']
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=plot_dpi, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        html_content += f"""
        <div class="figure">
            <img src="data:image/png;base64,{img_base64}" alt="{plant} Scree Plot">
            <p class="figure-caption">Figure: Scree Plot for {plant}</p>
        </div>
        """

        # Loadings plot
        fig = results['loadings_plot']
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=plot_dpi, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        html_content += f"""
        <div class="figure">
            <img src="data:image/png;base64,{img_base64}" alt="{plant} PC1 Loadings">
            <p class="figure-caption">Figure: PC1 Loadings for {plant}</p>
        </div>
        """

        # Loadings table
        loadings_df = results['loadings']
        html_content += "\n<h4>Loadings Coefficients</h4>\n"
        html_content += loadings_df.to_html(classes='dataframe')

    # Add VPI summary
    html_content += "\n<h2>3. VPI Summary</h2>\n"
    vpi_summary = processed_data.groupby(['Plant', 'Treatment'])[
        'VPI'].agg(['mean', 'std', 'count'])
    vpi_summary.columns = ['Mean', 'Std Dev', 'N']
    html_content += vpi_summary.to_html(classes='dataframe')

    # Add RFD summary
    html_content += "\n<h2>4. Relative Fold Reduction (RFD) Summary</h2>\n"
    rfd_data = processed_data[processed_data['Treatment'] != 'T1']
    rfd_summary = rfd_data.groupby(['Plant', 'Treatment'])[
        'RFD'].agg(['mean', 'std', 'count'])
    rfd_summary.columns = ['Mean', 'Std Dev', 'N']
    html_content += rfd_summary.to_html(classes='dataframe')

    # Add statistical results
    html_content += "\n<h2>5. Statistical Analysis Results</h2>\n"

    for key in sorted(anova_results.keys()):
        plant, variable = key.rsplit('_', 1)

        html_content += f"\n<h3>{plant} - {variable}</h3>\n"

        # ANOVA table
        html_content += "<h4>ANOVA Table</h4>\n"
        html_content += anova_results[key].to_html(
            classes='dataframe', index=False)

        # Tukey results
        if key in tukey_results:
            html_content += "<h4>Treatment Comparison (Tukey HSD)</h4>\n"
            html_content += tukey_results[key].to_html(
                classes='dataframe', index=False)

    # Close HTML
    html_content += """
        </div>
    </body>
    </html>
    """

    return html_content


def plot_individual_variable_chart(data, plant, variable, tukey_results=None, dpi=300):
    """
    Create individual bar chart for a specific plant and variable with letter groupings

    Parameters:
    -----------
    data : pd.DataFrame
        Processed data
    plant : str
        Plant name
    variable : str
        Variable name (DSI, AUDSC, RFVL, VPI, or RFD)
    tukey_results : dict, optional
        Dictionary of Tukey HSD results with letter groups
    dpi : int
        Plot resolution

    Returns:
    --------
    matplotlib.figure.Figure : Individual variable chart
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'

    # Filter data for specific plant
    plant_data = data[data['Plant'] == plant].copy()

    # For RFD, exclude T1
    if variable == 'RFD':
        plant_data = plant_data[plant_data['Treatment'] != 'T1']

    # Get treatment order
    treatment_order = sorted(plant_data['Treatment'].unique())

    # Calculate summary statistics
    summary = plant_data.groupby('Treatment')[variable].agg([
        'mean', 'sem']).reset_index()
    summary.columns = ['Treatment', 'Mean', 'SE']
    summary['Treatment'] = pd.Categorical(
        summary['Treatment'],
        categories=treatment_order,
        ordered=True
    )
    summary = summary.sort_values('Treatment')

    # Get letter groups if available
    letter_groups = {}
    key = f"{plant}_{variable}"
    if tukey_results and key in tukey_results:
        tukey_df = tukey_results[key]
        for _, row in tukey_df.iterrows():
            letter_groups[row['Treatment']] = row['Group']

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)

    # Create bar plot
    bars = sns.barplot(
        x='Treatment',
        y='Mean',
        hue='Treatment',
        data=summary,
        ax=ax,
        palette='Set2',
        legend=False
    )

    # Calculate y_max for proper spacing
    y_max = summary['Mean'].max() + summary['SE'].max() * 1.5
    ax.set_ylim(0, y_max + (y_max * 0.15))

    # Add error bars and letter groups
    for idx, row in summary.iterrows():
        # Error bars
        ax.errorbar(
            idx,
            row['Mean'],
            yerr=row['SE'],
            fmt='none',
            ecolor='black',
            capsize=5,
            linewidth=1.5
        )

        # Add letter group
        if row['Treatment'] in letter_groups:
            text_y = row['Mean'] + row['SE'] + (y_max * 0.03)
            ax.text(
                idx, text_y,
                str(letter_groups[row['Treatment']]
                    ).lower(),  # Lowercase display
                ha='center',
                va='bottom',
                fontsize=25,
                fontweight='bold',
                color='black',
                fontname='Times New Roman'
            )

    # Y-axis labels
    y_labels = {
        "AUDSC": "Percent-Days",
        "DSI": "Percent",
        "RFVL": "Viral Transcript",
        "VPI": "Protection Index Value",
        "RFD": "Fold Reduction"
    }

    # Formatting
    ax.set_xlabel("Treatment", fontsize=25, fontweight='bold')
    ax.set_ylabel(y_labels.get(variable, variable),
                  fontsize=25, fontweight='bold')
    ax.set_title(f"{plant} - {variable}", fontsize=28,
                 fontweight='bold', pad=20)
    # Set x-axis ticks before labels
    ax.set_xticks(range(len(treatment_order)))
    ax.set_xticklabels(treatment_order, rotation=45, ha='right', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=25)

    # Italicize binomial names
    binomial_plants = ["N. benthamiana", "A. thaliana", "S. lycopersicum"]
    if plant in binomial_plants:
        ax.set_title(f"{plant} - {variable}", fontsize=28, fontweight='bold',
                     fontstyle='italic', pad=20)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()

    return fig
