# generate_paradigm_diagram.py
# Generates a diagram comparing forecasting paradigms using a REAL prediction from the models.

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # *** THIS LINE IS THE FIX ***

# --- Local Imports ---
import config
from utils import ProbabilisticHead, nll, create_sequences, inverse_transform

def load_models_and_data():
    """Loads all necessary models and data for generating the diagram."""
    print("Loading models and test data...")
    # Load Transformer
    transformer_model = keras.models.load_model(os.path.join(config.MODEL_DIR, 'transformer_model.keras'))
    
    # Load Bayesian
    custom_objects = {'ProbabilisticHead': ProbabilisticHead, 'nll': nll}
    bayesian_model_dist = keras.models.load_model(os.path.join(config.MODEL_DIR, 'bayesian_model.keras'), custom_objects=custom_objects)
    
    # Prepare data
    df = pd.read_parquet(config.PROCESSED_DATA_PATH)
    n = len(df)
    train_df = df[0:int(n*0.7)]
    test_df = df[int(n*0.9):]
    
    scaler = MinMaxScaler().fit(train_df)
    target_idx = list(df.columns).index(config.TARGET_FEATURE)
    n_features = len(df.columns)
    
    X_test, y_test_scaled = create_sequences(scaler.transform(test_df), config.SEQ_LENGTH, target_idx)
    
    y_test_actual = inverse_transform(y_test_scaled, scaler, target_idx, n_features)
    
    return transformer_model, bayesian_model_dist, X_test, y_test_actual, scaler, target_idx, n_features

def generate_diagram_from_real_data(t_model, b_model, X_test, y_actual, scaler, target_idx, n_features):
    """Generates the diagram using a real prediction from the test set."""
    
    # --- Select a Specific Timestep for the Case Study ---
    case_study_index = 200 
    
    X_sample = X_test[case_study_index:case_study_index+1]
    y_true_sample = y_actual[case_study_index]
    
    print(f"Generating case study for test sample #{case_study_index}...")
    print(f"  -> True Price (p_t): {y_true_sample:.2f} €/MWh")
    
    # --- 1. Get Deterministic Transformer Prediction ---
    t_pred_scaled = t_model.predict(X_sample, verbose=0)
    t_pred_actual = inverse_transform(t_pred_scaled, scaler, target_idx, n_features)[0]
    print(f"  -> Transformer Point Forecast (ŷ_t): {t_pred_actual:.2f} €/MWh")
    
    # --- 2. Get Probabilistic Bayesian Prediction ---
    mc_means_scaled = []
    mc_stddevs_scaled = []
    
    inputs = keras.Input(shape=b_model.input_shape[1:])
    dist_output = b_model(inputs)
    mean_output = layers.Lambda(lambda t: t.mean())(dist_output)
    stddev_output = layers.Lambda(lambda t: t.stddev())(dist_output)
    prediction_model = keras.Model(inputs=inputs, outputs=[mean_output, stddev_output])

    for _ in range(config.MONTE_CARLO_SAMPLES):
        mean_pred, stddev_pred = prediction_model.predict(X_sample, verbose=0)
        mc_means_scaled.append(mean_pred[0, 0])
        mc_stddevs_scaled.append(stddev_pred[0, 0])
        
    mc_means_actual = inverse_transform(np.array(mc_means_scaled), scaler, target_idx, n_features)
    
    final_mean = np.mean(mc_means_actual)
    epistemic_std = np.std(mc_means_actual)
    
    aleatoric_var_scaled = np.mean(np.square(mc_stddevs_scaled))
    data_range = scaler.data_range_[target_idx]
    aleatoric_std = np.sqrt(aleatoric_var_scaled) * data_range
    
    total_std = np.sqrt(aleatoric_std**2 + epistemic_std**2)
    
    print(f"  -> Bayesian Mean Forecast (μ_t): {final_mean:.2f} €/MWh")
    print(f"  -> Bayesian Total Uncertainty (σ_t): {total_std:.2f} €/MWh")
    
    # --- 3. Generate the Plot ---
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.family': 'serif', 'font.size': 14})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle("Forecasting Paradigms: A Case Study from the Test Set", fontsize=20, fontweight='bold')
    
    price_range = np.linspace(y_true_sample - 30, y_true_sample + 30, 1000)
    
    ax1.set_title("A) Deterministic (Point) Forecast", fontsize=16)
    ax1.plot([t_pred_actual], [0.5], 's', color='darkorange', markersize=12, label=f'Transformer Forecast = {t_pred_actual:.2f}')
    ax1.axvline(y_true_sample, color='black', linestyle='--', linewidth=2, label=f'True Value = {y_true_sample:.2f}')
    ax1.annotate('', xy=(t_pred_actual, 0.45), xytext=(y_true_sample, 0.45), arrowprops=dict(arrowstyle='<->', color='gray', lw=2))
    ax1.text((t_pred_actual + y_true_sample) / 2, 0.38, 'Error', ha='center', color='gray')
    ax1.set_xlabel("Electricity Price (€/MWh)"); ax1.set_ylabel("Conceptual Representation")
    ax1.set_yticks([]); ax1.set_ylim(0, 1); ax1.legend(loc='upper left')

    ax2.set_title("B) Probabilistic (Density) Forecast", fontsize=16)
    pdf = norm.pdf(price_range, final_mean, total_std)
    ax2.plot(price_range, pdf, color='royalblue', linewidth=3, label='Predicted Distribution')
    ax2.fill_between(price_range, pdf, color='royalblue', alpha=0.2)
    ax2.axvline(final_mean, color='royalblue', linestyle=':', linewidth=2, label=f'Mean Forecast (μ) = {final_mean:.2f}')
    ax2.axvline(y_true_sample, color='black', linestyle='--', linewidth=2, label=f'True Value = {y_true_sample:.2f}')
    
    lower_bound = final_mean - 1.96 * total_std
    upper_bound = final_mean + 1.96 * total_std
    mask = (price_range >= lower_bound) & (price_range <= upper_bound)
    ax2.fill_between(price_range[mask], pdf[mask], color='skyblue', alpha=0.4, label=f'95% C.I. [{lower_bound:.2f}, {upper_bound:.2f}]')

    ax2.set_xlabel("Electricity Price (€/MWh)"); ax2.set_yticks([])
    ax2.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    diagram_path = os.path.join(config.REPORT_DIR, 'forecasting_paradigms.png')
    plt.savefig(diagram_path, dpi=config.PLOT_DPI)
    print(f"\nParadigm diagram based on real data saved to: {diagram_path}")
    plt.show()

if __name__ == '__main__':
    transformer_model, bayesian_model_dist, X_test, y_test_actual, scaler, target_idx, n_features = load_models_and_data()
    generate_diagram_from_real_data(transformer_model, bayesian_model_dist, X_test, y_test_actual, scaler, target_idx, n_features)
