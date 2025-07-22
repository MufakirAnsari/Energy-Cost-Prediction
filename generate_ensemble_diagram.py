# generate_ensemble_diagram.py

import os
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- Local Imports ---
import config

def generate_ensemble_tree_visualization():
    """
    Loads the trained LightGBM ensemble model and generates a high-quality,
    annotated visualization of one of its decision trees for the thesis.
    """
    print("--- Generating Publication-Quality Ensemble Tree Diagram ---")

    os.makedirs(config.REPORT_DIR, exist_ok=True)
    
    model_path = os.path.join(config.MODEL_DIR, 'ensemble_model.joblib')
    diagram_path = os.path.join(config.REPORT_DIR, 'diagram_ensemble_tree.png')

    try:
        model = joblib.load(model_path)
        print("Ensemble model loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Ensemble model not found at {model_path}.")
        print("Please run 'python 04_ensemble.py' first.")
        return

    # --- Plot Generation ---
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 14,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.figsize': (25, 18)
    })

    # Create the plot object
    ax = lgb.plot_tree(
        model, 
        tree_index=0,
        show_info=['split_gain', 'internal_value', 'internal_count', 'leaf_count'],
        precision=3
        # --- REMOVED THE INVALID ARGUMENT: tree_style='classic' ---
    )
    
    # --- Annotations and Aesthetics ---
    plt.title(
        "Visualization of a Representative Decision Tree from the LightGBM Ensemble",
        fontsize=24, 
        fontweight='bold', 
        pad=20
    )

    props = dict(boxstyle='round,pad=0.5', facecolor='aliceblue', alpha=0.9)
    
    interpretation_text = (
        "How to Interpret This Diagram:\n\n"
        "- Each box represents a decision node or a leaf.\n"
        "- The first line shows the splitting condition (e.g., 'bayesian_pred <= 0.456').\n"
        "- 'split_gain' is the reduction in variance achieved by this split.\n"
        "- 'internal_value' is the prediction made at this node.\n"
        "- 'internal_count' is the number of validation samples that pass through this node.\n\n"
        "This tree learns to correct the base models' predictions by making a series\n"
        "of decisions based on their output values."
    )

    ax.text(0.98, 0.98, interpretation_text, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # --- Save and Show ---
    plt.savefig(
        diagram_path, 
        dpi=config.PLOT_DPI, 
        bbox_inches='tight'
    )
    print(f"\nEnsemble tree diagram saved to: {diagram_path}")
    plt.show()

if __name__ == '__main__':
    generate_ensemble_tree_visualization()