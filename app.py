from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import os
from io import BytesIO
import base64
import matplotlib.pyplot as plt

# Import utility modules
from utils.data_processing import calculate_rfd, compute_vpi_pca
from utils.statistics import perform_anova_tukey
from utils.visualizations import (
    plot_scree_plot,
    plot_pca_loadings,
    plot_comparative_charts,
    plot_vpi_chart,
    plot_rfd_chart,
    plot_individual_variable_chart,
    generate_html_report
)

# Page configuration
st.set_page_config(
    page_title="VPI Analysis Platform",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
#fixed-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: {secondary_bg};
    border-top: 2px solid {border_color};
    padding: 0.8rem 1.5rem;
    z-index: 1000;
    font-family: "Times New Roman", serif;
}

#fixed-footer p {
    font-size: 0.85rem;
    color: {text_color};
    margin: 0;
    text-align: center;
}

#fixed-footer a {
    font-weight: bold;
    color: {border_color};
    margin: 0 0.4rem;
    text-decoration: none;
}

/* Footer links */
#fixed-footer a {
    font-weight: bold;
    color: #2E7D32;
    margin: 0 0.4rem;
}

#fixed-footer a:hover {
    text-decoration: underline;
    opacity: 0.85;
}

/* Prevent footer overlap */
.main > div {
    padding-bottom: 4rem;
}

/* Fix PDF visibility inside expander */
.streamlit-expanderContent {
    background-color: {card_bg} !important;
    padding: 1rem !important;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Theme-specific styles
if st.session_state.theme == 'light':
    # Light theme colors
    bg_color = "#FFFFFF"
    text_color = "#000000"
    secondary_bg = "#F8F9FA"
    border_color = "#2E7D32"
    card_bg = "#FFFFFF"
    card_text = "#000000"
    header_bg = "#E8F5E9"
    header_text = "#1B5E20"
    info_bg = "#E3F2FD"
    info_text = "#0D47A1"
    info_border = "#2196F3"
    metric_border = "#2E7D32"
    metric_label = "#2E7D32"
else:
    # Dark theme colors
    bg_color = "#0E1117"
    text_color = "#FAFAFA"
    secondary_bg = "#262730"
    border_color = "#66BB6A"
    card_bg = "#1E1E1E"
    card_text = "#FAFAFA"
    header_bg = "#2E7D32"
    header_text = "#E8F5E9"
    info_bg = "#1565C0"
    info_text = "#E3F2FD"
    info_border = "#42A5F5"
    metric_border = "#66BB6A"
    metric_label = "#81C784"

# Dynamic CSS based on theme - WITH DROPDOWN FIX
st.markdown(f"""
    <style>
    /* Main background */
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    
    /* Main header */
    .main-header {{
        font-size: 2.5rem;
        font-weight: bold;
        color: {header_text} !important;
        text-align: center;
        padding: 1rem;
        background-color: {header_bg} !important;
        border-radius: 10px;
        margin-bottom: 1rem;
    }}
    
    /* Top navigation bar */
    header {{
        background-color: {header_bg} !important;
    }}
    
    header * {{
        color: {header_text} !important;
    }}
    
    /* Streamlit header */
    [data-testid="stHeader"] {{
        background-color: {header_bg} !important;
    }}
    
    /* All text elements */
    .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span,
    p, span, div, label, li, h1, h2, h3, h4, h5 {{
        color: {text_color} !important;
    }}
    
    strong, b {{
        color: {text_color} !important;
        font-weight: 900 !important;
    }}
    
    /* Dataframes */
    .dataframe {{
        color: {text_color} !important;
        background-color: {card_bg} !important;
    }}
    
    .dataframe th {{
        background-color: {border_color} !important;
        color: {card_bg} !important;
        font-weight: bold !important;
    }}
    
    .dataframe td {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
    }}
    
    /* Dataframe toolbar buttons (download, search, fullscreen) */
    [data-testid="stDataFrameResizable"] button {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
    }}
    
    [data-testid="stDataFrameResizable"] button:hover {{
        background-color: {border_color} !important;
        color: {bg_color} !important;
    }}
    
    /* Dataframe toolbar */
    [data-testid="stDataFrameResizable"] [data-testid="stElementToolbar"] {{
        background-color: {card_bg} !important;
    }}
    
    [data-testid="stDataFrameResizable"] [data-testid="stElementToolbar"] button {{
        background-color: {secondary_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
    }}
    
    /* Search input in dataframe */
    [data-testid="stDataFrameResizable"] input {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
    }}
    
    /* Dataframe icons */
    [data-testid="stDataFrameResizable"] svg {{
        fill: {text_color} !important;
        stroke: {text_color} !important;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2rem;
        background-color: {secondary_bg};
    }}
    
    .stTabs [data-baseweb="tab"] {{
        font-size: 1.1rem;
        font-weight: 600;
        color: {text_color} !important;
        background-color: {secondary_bg};
    }}
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background-color: {border_color} !important;
        color: {bg_color} !important;
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {secondary_bg};
    }}
    
    [data-testid="stSidebar"] * {{
        color: {text_color} !important;
    }}
    
    /* Info boxes */
    .stAlert {{
        background-color: {info_bg} !important;
        color: {info_text} !important;
        border-left: 4px solid {info_border} !important;
    }}
    
    /* Success/Error/Warning boxes */
    .stSuccess, .stError, .stWarning, .stInfo {{
        color: {text_color} !important;
    }}
    
    /* Buttons */
    .stButton>button {{
        color: {text_color};
        border: 2px solid {border_color};
        font-weight: bold;
        background-color: {secondary_bg};
    }}
    
    .stButton>button:hover {{
        background-color: {border_color};
        color: {bg_color};
    }}
    
    /* Download buttons */
    .stDownloadButton>button {{
        background-color: {border_color};
        color: {bg_color};
        font-weight: bold;
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        color: {text_color} !important;
        background-color: {secondary_bg};
    }}
    
    /* Select boxes and inputs - LABELS */
    .stSelectbox label, .stSlider label, .stFileUploader label, .stTextInput label {{
        color: {text_color} !important;
        font-weight: bold !important;
    }}
    
    /* CRITICAL FIX - Select box dropdown container */
    .stSelectbox > div > div {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border: 2px solid {border_color} !important;
    }}
    
    /* Dropdown popover */
    [data-baseweb="popover"] {{
        background-color: {card_bg} !important;
    }}
    
    /* Dropdown menu container */
    [data-baseweb="menu"] {{
        background-color: {card_bg} !important;
    }}
    
    [data-baseweb="menu"] ul {{
        background-color: {card_bg} !important;
    }}
    
    /* Individual dropdown options */
    [role="option"] {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        padding: 8px 12px !important;
    }}
    
    [role="option"]:hover {{
        background-color: {border_color} !important;
        color: {bg_color} !important;
    }}
    
    /* Selected option in dropdown */
    [aria-selected="true"][role="option"] {{
        background-color: {border_color} !important;
        color: {bg_color} !important;
    }}
    
    /* FILE UPLOADER - ENHANCED VISIBILITY */
    [data-testid="stFileUploader"] {{
        background-color: {card_bg} !important;
        border: 2px dashed {border_color} !important;
        padding: 20px !important;
    }}
    
    [data-testid="stFileUploader"] section {{
        background-color: {card_bg} !important;
        border: 2px dashed {border_color} !important;
        padding: 20px !important;
    }}
    
    [data-testid="stFileUploader"] section > div {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
    }}
    
    /* Browse files button - CRITICAL FIX */
    [data-testid="stFileUploader"] button {{
        background-color: {border_color} !important;
        color: {bg_color} !important;
        font-weight: bold !important;
        border: 2px solid {border_color} !important;
        padding: 8px 16px !important;
    }}
    
    [data-testid="stFileUploader"] button:hover {{
        background-color: {header_text} !important;
        color: {bg_color} !important;
    }}
    
    /* File uploader text */
    [data-testid="stFileUploader"] small {{
        color: {text_color} !important;
        font-weight: bold !important;
    }}
    
    [data-testid="stFileUploader"] label {{
        color: {text_color} !important;
        font-weight: bold !important;
    }}
    
    /* Drag and drop text */
    .uploadedFile {{
        color: {text_color} !important;
    }}
    
    /* File uploader instructions */
    [data-testid="stFileUploader"] span {{
        color: {text_color} !important;
    }}
    
    /* Input fields */
    input {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
    }}
    
    /* Select box base */
    [data-baseweb="select"] {{
        background-color: {card_bg} !important;
    }}
    
    /* Metric values */
    [data-testid="stMetricValue"] {{
        color: {text_color} !important;
        font-size: 2rem !important;
        font-weight: bold !important;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {metric_label} !important;
        font-weight: bold !important;
    }}
    
    /* Code blocks */
    code {{
        background-color: {secondary_bg} !important;
        color: {text_color} !important;
    }}
    
    /* Links */
    a {{
        color: {border_color} !important;
    }}
    
    /* ADDITIONAL FILE UPLOADER FIXES */
    .stFileUploader {{
        background-color: {card_bg} !important;
    }}
    
    .stFileUploader > div {{
        background-color: {card_bg} !important;
        border: 2px dashed {border_color} !important;
    }}
    
    /* Upload button base */
    button[kind="secondary"] {{
        background-color: {border_color} !important;
        color: {bg_color} !important;
        border: 2px solid {border_color} !important;
    }}
    
    /* All text in file upload area */
    [data-testid="stFileUploader"] div {{
        color: {text_color} !important;
    }}
    
    [data-testid="stFileUploader"] p {{
        color: {text_color} !important;
    }}
    
    /* DATAFRAME CONTROLS - Enhanced */
    .stDataFrame {{
        background-color: {card_bg} !important;
    }}
    
    /* Dataframe container */
    [data-testid="stDataFrame"] {{
        background-color: {card_bg} !important;
    }}
    
    /* Element toolbar (contains download, search, fullscreen) */
    [data-testid="stElementToolbar"] {{
        background-color: {card_bg} !important;
        padding: 4px !important;
    }}
    
    [data-testid="stElementToolbar"] button {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
        padding: 4px 8px !important;
    }}
    
    [data-testid="stElementToolbar"] button:hover {{
        background-color: {border_color} !important;
        color: {bg_color} !important;
    }}
    
    /* Toolbar icons */
    [data-testid="stElementToolbar"] svg {{
        fill: {text_color} !important;
    }}
    
    /* Search box in toolbar */
    [data-testid="stElementToolbar"] input {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
    }}
    
    /* All dataframe buttons */
    [class*="dataframe"] button {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
    }}
    </style>
""", unsafe_allow_html=True)

# Helper function


def fig_to_bytes(fig, format='png', dpi=300):
    """Convert matplotlib figure to bytes for download"""
    buf = BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()


# Initialize session state
for key in ['data', 'processed_data', 'pca_results', 'anova_results', 'tukey_results']:
    if key not in st.session_state:
        st.session_state[key] = None

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/plant-under-sun.png", width=80)
    st.markdown("### VPI Analysis")

    # Theme toggle at the top of sidebar
    theme_label = "🌙 Dark Mode" if st.session_state.theme == 'light' else "☀️ Light Mode"
    if st.button(theme_label, use_container_width=True, key="theme_toggle"):
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
        st.rerun()

    st.markdown("---")

    # File upload
    st.markdown("#### Data Upload")
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload your plant disease data in CSV format"
    )

    # Sample data download
    st.markdown("#### Sample Data")
    if os.path.exists('data.csv'):
        with open('data.csv', 'r') as f:
            st.download_button(
                label="Download Sample Data",
                data=f.read(),
                file_name="sample_data.csv",
                mime="text/csv",
                use_container_width=True
            )

    st.markdown("---")

    # Column mapping information
    st.markdown("#### Expected Data Format")
    st.info("""
    **Default Column Order:**
    1. Plant (categorical)
    2. Treatment (T1, T2, T3, T4, T5)
    3. DSI (Disease Severity Index)
    4. AUDSC (Area Under Disease Severity Curve)
    5. RFVL (Relative Fold Viral Load)
    """)

    st.markdown("---")

    # Analysis settings
    st.markdown("#### Analysis Settings")
    alpha_level = st.selectbox(
        "Significance Level (α)",
        options=[0.01, 0.05, 0.10],
        index=1,
        help="Statistical significance level"
    )

    plot_dpi = st.slider(
        "Plot Resolution (DPI)",
        min_value=150,
        max_value=600,
        value=300,
        step=50
    )

    # Research paper link
    st.markdown("---")
    st.markdown("#### Research Paper")
    st.markdown("""
        
    **Citation:**  
    *Coming soon*
    """)

# Main content
st.markdown('<p class="main-header">Virus Protection Index (VPI) Analysis Platform</p>',
            unsafe_allow_html=True)

# Create tabs
tabs = st.tabs([
    "About",
    "Data Overview",
    "PCA Analysis",
    "Statistical Analysis",
    "VPI & RFD Results",
    "Export Results"
])

# Tab 1: About
with tabs[0]:
    st.markdown("## About PI Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Overview
        The **Virus Protection Index (VPI)** is a comprehensive metric 
        developed to assess the effectiveness of various treatments against plant viral infections.
        
        ### Key Features
        - **Multi-parameter assessment**: Integrates Disease Severity Index (DSI) and Relative Fold Viral Load (RFVL)
        - **PCA-based computation**: Uses Principal Component Analysis for robust index calculation
        - **Statistical validation**: ANOVA and Tukey HSD for treatment comparison
        - **Relative Fold Reduction (RFD)**: Quantifies viral load reduction compared to control
        
        ### Methodology
        
        #### 1. VPI Computation
        The VPI is calculated using Principal Component Analysis:
        
        1. **Feature Standardization**: DSI and RFVL are normalized using Min-Max scaling
        2. **PCA Application**: Extract principal components from standardized features
        3. **Index Calculation**: GVPI = 1 - normalized(PC1)
        
        Higher VPI values indicate better protection against the virus.
        
        #### 2. RFD Calculation
        Relative Fold Reduction measures viral load reduction:
        
        RFD = RFVL(Control) / RFVL(Treatment)
        
        Higher RFD values indicate greater viral load reduction.
        
        #### 3. Statistical Analysis
        - **ANOVA**: Tests for significant differences between treatments
        - **Tukey HSD**: Post-hoc pairwise comparisons with letter grouping
        - **Critical Difference (CD)**: Minimum difference for significance at α=0.05        
        """)
    with col2:
        st.markdown("### Analysis Workflow")
        workflow_steps = [
            "1️. Upload Data",
            "2️. Data Validation",
            "3️. PCA Analysis",
            "4️. VPI Computation",
            "5️. RFD Calculation",
            "6️. Statistical Tests",
            "7️. Visualization",
            "8️. Export Results"
        ]

        for step in workflow_steps:
            st.markdown(f"**{step}**")

# Tab 2: Data Overview
with tabs[1]:
    st.markdown("## Data Overview and Validation")

    if uploaded_file is not None:
        try:
            # Load data
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data

            # Validation
            required_cols = ['Plant', 'Treatment', 'DSI', 'AUDSC', 'RFVL']
            missing_cols = [
                col for col in required_cols if col not in data.columns]

            if missing_cols:
                st.error(
                    f"Missing required columns: {', '.join(missing_cols)}")
                st.stop()

            st.success("Data loaded successfully!")

            # Display statistics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f'''<div style="background-color: {card_bg}; padding: 1.5rem; border-radius: 0.5rem; border: 3px solid {metric_border}; text-align: center;">
                    <p style="color: {metric_label}; font-size: 1rem; font-weight: bold; margin: 0;">📊 Total Observations</p>
                    <p style="color: {card_text}; font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0 0 0;">{len(data)}</p>
                </div>''', unsafe_allow_html=True)
            with col2:
                st.markdown(f'''<div style="background-color: {card_bg}; padding: 1.5rem; border-radius: 0.5rem; border: 3px solid {metric_border}; text-align: center;">
                    <p style="color: {metric_label}; font-size: 1rem; font-weight: bold; margin: 0;">🌱 Plant Types</p>
                    <p style="color: {card_text}; font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0 0 0;">{data["Plant"].nunique()}</p>
                </div>''', unsafe_allow_html=True)
            with col3:
                st.markdown(f'''<div style="background-color: {card_bg}; padding: 1.5rem; border-radius: 0.5rem; border: 3px solid {metric_border}; text-align: center;">
                    <p style="color: {metric_label}; font-size: 1rem; font-weight: bold; margin: 0;">🧬 Treatments</p>
                    <p style="color: {card_text}; font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0 0 0;">{data["Treatment"].nunique()}</p>
                </div>''', unsafe_allow_html=True)
            with col4:
                st.markdown(f'''<div style="background-color: {card_bg}; padding: 1.5rem; border-radius: 0.5rem; border: 3px solid {metric_border}; text-align: center;">
                    <p style="color: {metric_label}; font-size: 1rem; font-weight: bold; margin: 0;">📈 Variables</p>
                    <p style="color: {card_text}; font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0 0 0;">3</p>
                </div>''', unsafe_allow_html=True)

            st.markdown("---")

            # Data preview
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("### Data Preview")
                st.dataframe(
                    data.head(20), use_container_width=True, height=400)

            with col2:
                st.markdown("### Plants in Dataset")
                for plant in data['Plant'].unique():
                    count = len(data[data['Plant'] == plant])
                    st.markdown(f"**• {plant}**: {count} obs")

                st.markdown("### Treatments")
                for treatment in sorted(data['Treatment'].unique()):
                    count = len(data[data['Treatment'] == treatment])
                    st.markdown(f"**• {treatment}**: {count} obs")

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.stop()
    else:
        st.info("Please upload a CSV file using the sidebar to begin analysis")

# Tab 3: PCA Analysis
with tabs[2]:
    st.markdown("## Principal Component Analysis")

    if st.session_state.data is not None:
        if st.button("Run PCA Analysis", type="primary", use_container_width=True):
            with st.spinner("Performing PCA analysis..."):
                processed_data, pca_results = compute_vpi_pca(
                    st.session_state.data, plot_dpi)
                st.session_state.processed_data = processed_data
                st.session_state.pca_results = pca_results
                st.success("PCA Analysis completed!")

        if st.session_state.pca_results:
            plants = st.session_state.data['Plant'].unique()

            for plant in plants:
                st.markdown(f"### {plant}")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Scree Plot")
                    scree_fig = st.session_state.pca_results[plant]['scree_plot']
                    st.pyplot(scree_fig)

                    # Download button
                    st.download_button(
                        label="Download Scree Plot",
                        data=fig_to_bytes(scree_fig, dpi=plot_dpi),
                        file_name=f"{plant}_scree_plot.png",
                        mime="image/png",
                        key=f"scree_{plant}"
                    )

                    # Explained variance - Theme-aware
                    exp_var = st.session_state.pca_results[plant]['explained_variance']
                    st.markdown(f"""
                    <div style="background-color: {card_bg}; padding: 1.2rem; border-radius: 0.5rem; border: 3px solid {metric_border}; margin-top: 1rem;">
                    <p style="color: {card_text}; font-size: 1.1rem; margin: 0;"><strong>PC1 Explained Variance:</strong> {exp_var[0]*100:.2f}%</p>
                    <p style="color: {card_text}; font-size: 1.1rem; margin: 0.5rem 0 0 0;"><strong>PC2 Explained Variance:</strong> {exp_var[1]*100:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown("#### PC1 Loadings")
                    loadings_fig = st.session_state.pca_results[plant]['loadings_plot']
                    st.pyplot(loadings_fig)

                    # Download button
                    st.download_button(
                        label="Download Loadings Plot",
                        data=fig_to_bytes(loadings_fig, dpi=plot_dpi),
                        file_name=f"{plant}_loadings_plot.png",
                        mime="image/png",
                        key=f"loadings_{plant}"
                    )

                    # Loadings table
                    loadings = st.session_state.pca_results[plant]['loadings']
                    st.dataframe(loadings, use_container_width=True)

                st.markdown("---")
    else:
        st.warning("Please upload data in the 'Data Overview' tab first")

# Tab 4: Statistical Analysis - WITH INDIVIDUAL CHARTS
with tabs[3]:
    st.markdown("## Statistical Analysis (ANOVA & Tukey HSD)")

    if st.session_state.processed_data is not None:
        if st.button("Run Statistical Analysis", type="primary", use_container_width=True):
            with st.spinner("Performing ANOVA and Tukey HSD tests..."):
                anova_results, tukey_results = perform_anova_tukey(
                    st.session_state.processed_data,
                    alpha=alpha_level
                )
                st.session_state.anova_results = anova_results
                st.session_state.tukey_results = tukey_results
                st.success("Statistical Analysis completed!")

        if st.session_state.anova_results:
            # Comparative charts with letter groupings
            st.markdown("### Comparative Analysis Charts")

            if st.button("Generate Comparative Charts (DSI, AUDSC, RFVL)", use_container_width=True):
                with st.spinner("Creating comparative charts..."):
                    comp_fig = plot_comparative_charts(
                        st.session_state.processed_data,
                        st.session_state.tukey_results,
                        plot_dpi
                    )
                    st.pyplot(comp_fig)

                    st.download_button(
                        label="Download Comparative Charts",
                        data=fig_to_bytes(comp_fig, dpi=plot_dpi),
                        file_name="comparative_charts.png",
                        mime="image/png",
                        key="comp_chart_dl"
                    )

            st.markdown("---")

            # ANOVA and Tukey results with INDIVIDUAL CHART
            st.markdown("### View Statistical Results by Variable")

            col1, col2 = st.columns(2)

            with col1:
                selected_plant = st.selectbox(
                    "Select Plant",
                    options=st.session_state.data['Plant'].unique(),
                    key="stat_plant_select"
                )

            with col2:
                selected_variable = st.selectbox(
                    "Select Variable",
                    options=['DSI', 'AUDSC', 'RFVL', 'VPI', 'RFD'],
                    key="stat_var_select"
                )

            # INDIVIDUAL BAR CHART WITH ERROR BARS AND LETTERS
            st.markdown(
                f"### Bar Chart: {selected_plant} - {selected_variable}")

            if st.button("Generate Individual Chart with Letters", use_container_width=True, key="gen_ind_chart"):
                with st.spinner(f"Creating chart for {selected_plant} - {selected_variable}..."):
                    ind_fig = plot_individual_variable_chart(
                        st.session_state.processed_data,
                        selected_plant,
                        selected_variable,
                        st.session_state.tukey_results,
                        plot_dpi
                    )
                    st.pyplot(ind_fig)

                    st.download_button(
                        label="Download Individual Chart",
                        data=fig_to_bytes(ind_fig, dpi=plot_dpi),
                        file_name=f"{selected_plant}_{selected_variable}_chart.png",
                        mime="image/png",
                        key="ind_chart_dl"
                    )

            st.markdown("---")

            # Display ANOVA results
            st.markdown(
                f"### ANOVA Results: {selected_plant} - {selected_variable}")

            key = f"{selected_plant}_{selected_variable}"
            if key in st.session_state.anova_results:
                anova_df = st.session_state.anova_results[key]
                st.dataframe(anova_df, use_container_width=True)

                # Interpretation
                p_value = anova_df[anova_df['SoV'] ==
                                   'Treatment']['p-value'].values[0]
                if p_value < alpha_level:
                    st.success(
                        f"Significant difference detected (p = {p_value:.4f}, α = {alpha_level})")
                else:
                    st.info(
                        f"No significant difference (p = {p_value:.4f}, α = {alpha_level})")

            st.markdown("---")

            # Tukey results
            st.markdown(
                f"### Tukey HSD Results: {selected_plant} - {selected_variable}")

            if key in st.session_state.tukey_results:
                tukey_df = st.session_state.tukey_results[key]
                st.dataframe(tukey_df, use_container_width=True)

                st.info("""
                **Interpretation:** Treatments with the same letter are NOT significantly different.
                - Single letter (e.g., "a", "b") = treatment belongs to one group only
                - Multiple letters (e.g., "ab", "abc") = treatment belongs to multiple overlapping groups
                - Example: If T3="ab", it means T3 is not different from treatments in group "a" AND not different from those in group "b"
                
                Group 'a' = highest mean, 'b' = second highest, etc.
                """)
    else:
        st.warning("Please complete PCA Analysis first")

# Tab 5: vPI & RFD Results
with tabs[4]:
    st.markdown("## VPI and RFD Results")

    if st.session_state.processed_data is not None:
        processed_data = st.session_state.processed_data

        # VPI Summary
        st.markdown("### VPI Summary by Treatment")
        vpi_summary = processed_data.groupby(['Plant', 'Treatment'])[
            'VPI'].agg(['mean', 'std', 'count'])
        vpi_summary.columns = ['Mean VPI', 'Std Dev', 'N']
        st.dataframe(vpi_summary.style.format({'Mean VPI': '{:.4f}', 'Std Dev': '{:.4f}'}),
                     use_container_width=True)

        st.markdown("---")

        # RFD Summary
        st.markdown("### Relative Fold Reduction (RFD) Summary")
        rfd_summary = processed_data[processed_data['Treatment'] != 'T1'].groupby(
            ['Plant', 'Treatment'])['RFD'].agg(['mean', 'std', 'count'])
        rfd_summary.columns = ['Mean RFD', 'Std Dev', 'N']
        st.dataframe(rfd_summary.style.format({'Mean RFD': '{:.4f}', 'Std Dev': '{:.4f}'}),
                     use_container_width=True)

        st.markdown("---")

        # Visualizations
        st.markdown("### Visualizations")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Generate VPI Chart", use_container_width=True):
                with st.spinner("Creating VPI chart..."):
                    tukey_res = st.session_state.tukey_results if st.session_state.tukey_results else None
                    vpi_fig = plot_vpi_chart(
                        processed_data, tukey_res, plot_dpi)
                    st.pyplot(vpi_fig)

                    st.download_button(
                        label="Download VPI Chart",
                        data=fig_to_bytes(vpi_fig, dpi=plot_dpi),
                        file_name="vpi_chart.png",
                        mime="image/png",
                        key="vpi_chart_dl"
                    )

        with col2:
            if st.button("Generate RFD Chart", use_container_width=True):
                with st.spinner("Creating RFD chart..."):
                    tukey_res = st.session_state.tukey_results if st.session_state.tukey_results else None
                    rfd_fig = plot_rfd_chart(
                        processed_data, tukey_res, plot_dpi)
                    st.pyplot(rfd_fig)

                    st.download_button(
                        label="Download RFD Chart",
                        data=fig_to_bytes(rfd_fig, dpi=plot_dpi),
                        file_name="rfd_chart.png",
                        mime="image/png",
                        key="rfd_chart_dl"
                    )
    else:
        st.warning("Please run PCA Analysis first")

# Tab 6: Export Results
with tabs[5]:
    st.markdown("## Export Results")

    if st.session_state.processed_data is not None:
        st.markdown("### Download Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Processed Data")
            csv = st.session_state.processed_data.to_csv(index=False)
            st.download_button(
                label="Download Processed Data (CSV)",
                data=csv,
                file_name="processed_data_with_vpi_rfd.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            if st.session_state.anova_results:
                st.markdown("#### ANOVA Results")
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    for key, df in st.session_state.anova_results.items():
                        df.to_excel(writer, sheet_name=key[:31], index=False)

                st.download_button(
                    label="Download ANOVA Results (Excel)",
                    data=output.getvalue(),
                    file_name="anova_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

        with col3:
            if st.session_state.tukey_results:
                st.markdown("#### Tukey Results")
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    for key, df in st.session_state.tukey_results.items():
                        df.to_excel(writer, sheet_name=key[:31], index=False)

                st.download_button(
                    label="Download Tukey Results (Excel)",
                    data=output.getvalue(),
                    file_name="tukey_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

        st.markdown("---")

        # Generate HTML Report
        st.markdown("### Comprehensive Analysis Report")

        if st.button("Generate HTML Report", type="primary", use_container_width=True):
            if all([st.session_state.pca_results, st.session_state.anova_results, st.session_state.tukey_results]):
                with st.spinner("Generating comprehensive HTML report..."):
                    html_report = generate_html_report(
                        st.session_state.data,
                        st.session_state.processed_data,
                        st.session_state.pca_results,
                        st.session_state.anova_results,
                        st.session_state.tukey_results,
                        plot_dpi
                    )

                    st.download_button(
                        label="Download HTML Report",
                        data=html_report,
                        file_name="vpi_analysis_report.html",
                        mime="text/html",
                        use_container_width=True
                    )
                    st.success(
                        "HTML Report generated! Click the button above to download.")
            else:
                st.warning(
                    "Please complete all analyses (PCA and Statistical Analysis) before generating the report.")
    else:
        st.warning(
            "No results available for export. Please complete the analysis first.")

st.markdown("""
<div id="fixed-footer">
    <p style="text-align:center;">
        <strong>VPI Analysis Platform</strong> · Version 1.0 · © 2025 &nbsp;|&nbsp;
        <a href="https://scholar.google.com/citations?user=qVfwLrYAAAAJ&hl=en&oi=ao" target="_blank">
            Dr. Suryakant Manik
        </a> ·
        <a href="https://scholar.google.com/citations?user=Es-kJk4AAAAJ&hl=en" target="_blank">
            Dr. Sandip Garai
        </a> ·
        <a href="https://scholar.google.com/citations?user=0dQ7Sf8AAAAJ&hl=en&oi=ao" target="_blank">
            Dr. Kanaka K K
        </a> ·
        <a href="https://www.researchgate.net/profile/Suman-Dutta-7" target="_blank">
            Dr. Suman Dutta
        </a>
        &nbsp;|&nbsp;
        <a href="mailto:drgaraislab@gmail.com">Contact</a>
    </p>
</div>
""", unsafe_allow_html=True)
