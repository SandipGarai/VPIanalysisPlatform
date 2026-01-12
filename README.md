# GBNVPI Analysis Platform

## Overview

The **Groundnut Bud Necrosis Virus Protection Index (GBNVPI) Analysis Platform** is a publication-ready Streamlit application for comprehensive plant disease analysis. This tool integrates Principal Component Analysis (PCA), statistical testing (ANOVA and Tukey HSD), and publication-quality visualizations to assess treatment effectiveness against viral infections in plants.

## Features

### Core Analysis

- **PCA-based GBNVPI Computation**: Calculates a protection index from Disease Severity Index (DSI) and Relative Fold Viral Load (RFVL)
- **Relative Fold Reduction (RFD)**: Quantifies viral load reduction compared to control treatments
- **Statistical Analysis**: ANOVA and Tukey HSD post-hoc tests for treatment comparisons
- **Publication-Quality Visualizations**: High-resolution plots ready for scientific publications

### User Interface

- **Multi-Tab Interface**: Organized workflow from data upload to results export
- **Interactive Data Exploration**: Real-time data validation and preview
- **Customizable Analysis**: Adjustable significance levels and plot parameters
- **Sample Data Included**: Download template data to get started quickly

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download the repository**

```bash
git clone <repository-url>
cd VPIanalysisPlatform
```

2. **Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

1. **Start the Streamlit app**

```bash
streamlit run app.py
```

2. **Access the application**

- The app will automatically open in your default browser
- Default URL: `http://localhost:8501`

### Data Format

Your CSV file should have the following columns in this order:

| Plant  | Treatment | DSI   | AUDSC  | RFVL |
| ------ | --------- | ----- | ------ | ---- |
| Cowpea | T1        | 95.83 | 514.58 | 1.43 |
| Cowpea | T2        | 29.17 | 125.00 | 0.34 |
| ...    | ...       | ...   | ...    | ...  |

**Column Descriptions:**

- **Plant**: Plant species name (categorical)
- **Treatment**: Treatment code (T1, T2, T3, T4, T5, etc.)
  - T1 is typically the control/infected treatment
- **DSI**: Disease Severity Index (0-100%)
- **AUDSC**: Area Under Disease Severity Curve (percent-days)
- **RFVL**: Relative Fold Viral Load (from qRT-PCR)

### Workflow

1. **About Tab**: Read methodology and understand the analysis
2. **Data Overview Tab**:
   - Upload your CSV file
   - Validate data structure
   - Preview statistics
3. **PCA Analysis Tab**:
   - Run PCA analysis
   - View scree plots and loadings
   - GBNVPI automatically calculated
4. **GBNVPI Results Tab**:
   - View GBNVPI and RFD summaries
   - Generate comparative charts
5. **Statistical Analysis Tab**:
   - Run ANOVA and Tukey HSD tests
   - View treatment groupings
   - Interpret significance
6. **Export Results Tab**:
   - Download processed data
   - Export statistical results
   - Generate comprehensive reports

## Methodology

### GBNVPI Calculation

The GBNVPI is computed using the following steps:

1. **Feature Standardization**: DSI and RFVL are normalized using Min-Max scaling (0-1 range)
2. **PCA Application**: Principal components are extracted from standardized features
3. **Index Calculation**:
   ```
   GBNVPI = 1 - normalized(PC1)
   ```
4. **Interpretation**: Higher GBNVPI values indicate better protection against the virus

### RFD Calculation

Relative Fold Reduction quantifies viral load reduction:

```
RFD = RFVL(Control T1) / RFVL(Treatment)
```

Higher RFD values indicate greater viral load reduction compared to control.

### Statistical Analysis

- **ANOVA**: One-way ANOVA tests for significant differences between treatments
- **Tukey HSD**: Post-hoc pairwise comparisons with letter grouping
- **Critical Difference (CD)**: Minimum difference for significance at α=0.05

## Configuration

### Analysis Settings (Sidebar)

- **Significance Level (α)**: Choose from 0.01, 0.05, or 0.10
- **Plot Resolution (DPI)**: Adjust from 150-600 for publication quality

### Expected Plant Nomenclature

The application handles binomial nomenclature correctly:

- Binomial names (e.g., _N. benthamiana_) are italicized in plots
- Common names (e.g., Cowpea) are displayed normally

## Output Files

### Processed Data CSV

- Original data plus GBNVPI and RFD columns
- Ready for further analysis or publication

### ANOVA Results (Excel)

- Separate sheets for each Plant × Variable combination
- Includes F-statistics, p-values, SEM, and Critical Difference

### Tukey HSD Results (Excel)

- Treatment means with standard errors
- Letter groupings for significance
- Sorted by mean values

### Plots (High-Resolution)

- Scree plots for each plant
- PC1 loadings charts
- Comparative bar charts
- GBNVPI and RFD comparison plots

## Troubleshooting

### Common Issues

**Issue: Missing columns error**

- Solution: Ensure your CSV has exactly these columns: Plant, Treatment, DSI, AUDSC, RFVL

**Issue: Plots not displaying**

- Solution: Click the analysis buttons in sequence (PCA Analysis → Generate Charts)

**Issue: Statistical analysis fails**

- Solution: Ensure you have at least 3 replications per treatment

**Issue: RFD values are NaN**

- Solution: Check that T1 (control) treatment is present for each plant

## Citation

If you use this tool in your research, please cite:

```
[Coming soon]
```

## Support

For questions, issues, or feature requests:

- Email: [sandipnicksandy@gmail.com]

## License

[Add your license information here]

## Acknowledgments

This application was developed to support plant disease research and facilitate reproducible analysis of treatment efficacy against viral infections.

## Version History

- **v1.0.0** (2025): Initial release
  - PCA-based GBNVPI computation
  - ANOVA and Tukey HSD analysis
  - Publication-quality visualizations
  - Multi-tab Streamlit interface

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

**Developed for Plant Disease Research | Publication-Ready Analysis Platform**

