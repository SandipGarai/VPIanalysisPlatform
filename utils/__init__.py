"""
vpi Analysis Utilities
"""

from .data_processing import calculate_rfd, compute_vpi_pca, validate_data
from .statistics import perform_anova_tukey, critical_difference, assign_letter_groups
from .visualizations import (
    plot_scree_plot,
    plot_pca_loadings,
    plot_comparative_charts,
    plot_vpi_chart,
    plot_rfd_chart,
    plot_individual_variable_chart,
    generate_html_report
)

__all__ = [
    'calculate_rfd',
    'compute_vpi_pca',
    'validate_data',
    'perform_anova_tukey',
    'critical_difference',
    'assign_letter_groups',
    'plot_scree_plot',
    'plot_pca_loadings',
    'plot_comparative_charts',
    'plot_vpi_chart',
    'plot_rfd_chart',
    'plot_individual_variable_chart',
    'generate_html_report'
]
