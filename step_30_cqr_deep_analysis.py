import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, anderson_ksamp
import config
import warnings

# Suppress runtime warnings from anderson_ksamp
warnings.filterwarnings("ignore", category=UserWarning)

def calculate_picp(y_true, lower, upper):
    """Calculate Prediction Interval Coverage Probability."""
    return np.mean((y_true >= lower) & (y_true <= upper))

def main():
    """
    Generate deeper analysis of CQR limitations requested by Reviewer #5.
    Produces formal exchangeability tests, time-varying coverage plots,
    and visual distribution comparisons of calibration vs test sets.
    """
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    fig_dir = os.path.join(config.REPORT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("Starting CQR deep analysis...")
    exchangeability_results = []
    
    # Data dictionaries to hold stuff for plotting outside the loop
    plot_data = {
        "PJM": {},
        "ERCOT": {}
    }
    
    for market in ["PJM", "ERCOT"]:
        print(f"\n--- Processing {market} ---")
        
        # Load Data
        cal_path = config.PJM_CAL_PATH if market == "PJM" else config.ERCOT_CAL_PATH
        test_path = config.PJM_TEST_PATH if market == "PJM" else config.ERCOT_TEST_PATH
        
        print(f"Loading {market} calibration and test sets...")
        cal_df = pd.read_parquet(cal_path)
        test_df = pd.read_parquet(test_path)
        
        X_cal = cal_df.drop(columns=[config.TARGET_COL])
        y_cal = cal_df[config.TARGET_COL].values
        
        X_test = test_df.drop(columns=[config.TARGET_COL])
        y_test = test_df[config.TARGET_COL].values
        
        # Load LightGBM Point model to get residuals
        print(f"Computing residuals for {market}...")
        lgbm_point_path = os.path.join(config.MODEL_DIR, f"lgbm_point_{market.lower()}.joblib")
        lgbm_point_model = joblib.load(lgbm_point_path)
        
        cal_preds = lgbm_point_model.predict(X_cal)
        test_preds = lgbm_point_model.predict(X_test)
        
        cal_residuals = y_cal - cal_preds
        test_residuals = y_test - test_preds
        
        # Store for distribution plot
        plot_data[market]['y_cal'] = y_cal
        plot_data[market]['y_test'] = y_test
        plot_data[market]['res_cal'] = cal_residuals
        plot_data[market]['res_test'] = test_residuals
        
        # Output 1: Formal Exchangeability Test
        print(f"Running exchangeability tests for {market}...")
        ks_stat, ks_pval = ks_2samp(cal_residuals, test_residuals)
        
        try:
            ad_res = anderson_ksamp([cal_residuals, test_residuals])
            ad_stat = ad_res.statistic
            ad_pval = ad_res.pvalue
        except Exception as e:
            print(f"Warning: Anderson-Darling test failed: {e}")
            ad_stat = np.nan
            ad_pval = np.nan
            
        exchangeability_results.append({
            "Market": market,
            "KS_Statistic": ks_stat,
            "KS_pvalue": ks_pval,
            "AD_Statistic": ad_stat,
            "AD_pvalue": ad_pval
        })
        
        # Read Predictions for Time-Varying Coverage
        print(f"Computing monthly PICP for {market}...")
        cqr_path = os.path.join(config.REPORT_DIR, f"cqr_preds_{market.lower()}.csv")
        qrf_path = os.path.join(config.REPORT_DIR, f"qrf_preds_{market.lower()}.csv")
        
        cqr_df = pd.read_csv(cqr_path, parse_dates=["datetime_utc"])
        qrf_df = pd.read_csv(qrf_path, parse_dates=["datetime_utc"])
        
        cqr_df.set_index("datetime_utc", inplace=True)
        qrf_df.set_index("datetime_utc", inplace=True)
        
        monthly_cqr = []
        monthly_qrf = []
        monthly_lgbm = []
        months = []
        
        for name, group in cqr_df.resample("ME"):
            if len(group) == 0:
                continue
            months.append(name)
            picp_cqr = calculate_picp(group["actual"], group["cqr_lower"], group["cqr_upper"])
            picp_lgbm = calculate_picp(group["actual"], group["raw_q05"], group["raw_q95"])
            monthly_cqr.append(picp_cqr)
            monthly_lgbm.append(picp_lgbm)
            
        for name, group in qrf_df.resample("ME"):
            if len(group) == 0:
                continue
            picp_qrf = calculate_picp(group["actual"], group["q05"], group["q95"])
            monthly_qrf.append(picp_qrf)
            
        plot_data[market]['months'] = months
        plot_data[market]['monthly_cqr'] = monthly_cqr
        plot_data[market]['monthly_qrf'] = monthly_qrf
        plot_data[market]['monthly_lgbm'] = monthly_lgbm

    # Save Output 1: Exchangeability Test Table
    print("\nSaving exchangeability test results...")
    ex_df = pd.DataFrame(exchangeability_results)
    out1_path = os.path.join(config.REPORT_DIR, "table_exchangeability_test.csv")
    ex_df.to_csv(out1_path, index=False)
    print(f"Saved -> {out1_path}")

    # Output 2: Time-Varying Coverage Plot (Combined)
    print("Generating time-varying coverage plot...")
    plt.style.use(config.PLOT_STYLE)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    for i, market in enumerate(["PJM", "ERCOT"]):
        ax = axes[i]
        months = plot_data[market]['months']
        ax.plot(months, plot_data[market]['monthly_cqr'], marker="o", label="CQR", color=config.MODEL_COLORS.get("CQR", "#d62728"))
        ax.plot(months, plot_data[market]['monthly_qrf'], marker="s", label="QRF", color=config.MODEL_COLORS.get("QRF", "#bcbd22"))
        ax.plot(months, plot_data[market]['monthly_lgbm'], marker="^", label="LightGBM Quantile", color=config.MODEL_COLORS.get("LightGBM", "#2ca02c"))
        
        ax.axhline(y=0.90, color="black", linestyle="--", linewidth=2, label="Target 90%")
        ax.set_title(f"{market}: Time-Varying Interval Coverage (2024-2025)", fontsize=14)
        ax.set_ylabel("PICP (Coverage)", fontsize=12)
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, alpha=0.5)
        if i == 0:
            ax.legend(fontsize=11)
            
    axes[1].set_xlabel("Test Period (Months)", fontsize=12)
    plt.tight_layout()
    out2_path = os.path.join(fig_dir, "Fig_TimeVarying_Coverage.png")
    plt.savefig(out2_path, dpi=config.PLOT_DPI)
    plt.close()
    print(f"Saved -> {out2_path}")

    # Output 3: Calibration-Test Distribution Comparison (2x2)
    print("Generating distribution shift plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Limits for x-axis to clip extreme outliers for better visualization
    # Optional: Set reasonable limits based on 1st/99th percentiles, but KDE usually handles it
    
    # Top Row: PJM
    sns.kdeplot(plot_data['PJM']['y_cal'], ax=axes[0,0], label="Calibration (2022)", fill=True, color="blue", alpha=0.3)
    sns.kdeplot(plot_data['PJM']['y_test'], ax=axes[0,0], label="Test (2024-2025)", fill=True, color="red", alpha=0.3)
    axes[0,0].set_title("PJM: Price Distribution Shift", fontsize=14)
    axes[0,0].set_xlabel("Price ($/MWh)")
    axes[0,0].legend()
    axes[0,0].set_xlim(-50, 300)
    
    sns.kdeplot(plot_data['PJM']['res_cal'], ax=axes[0,1], label="Calibration Residuals", fill=True, color="blue", alpha=0.3)
    sns.kdeplot(plot_data['PJM']['res_test'], ax=axes[0,1], label="Test Residuals", fill=True, color="red", alpha=0.3)
    axes[0,1].set_title("PJM: Residual Distribution Shift", fontsize=14)
    axes[0,1].set_xlabel("Residual ($/MWh)")
    axes[0,1].legend()
    axes[0,1].set_xlim(-150, 150)
    
    # Bottom Row: ERCOT
    sns.kdeplot(plot_data['ERCOT']['y_cal'], ax=axes[1,0], label="Calibration (2022)", fill=True, color="blue", alpha=0.3)
    sns.kdeplot(plot_data['ERCOT']['y_test'], ax=axes[1,0], label="Test (2024-2025)", fill=True, color="red", alpha=0.3)
    axes[1,0].set_title("ERCOT: Price Distribution Shift", fontsize=14)
    axes[1,0].set_xlabel("Price ($/MWh)")
    axes[1,0].legend()
    axes[1,0].set_xlim(-50, 400)
    
    sns.kdeplot(plot_data['ERCOT']['res_cal'], ax=axes[1,1], label="Calibration Residuals", fill=True, color="blue", alpha=0.3)
    sns.kdeplot(plot_data['ERCOT']['res_test'], ax=axes[1,1], label="Test Residuals", fill=True, color="red", alpha=0.3)
    axes[1,1].set_title("ERCOT: Residual Distribution Shift", fontsize=14)
    axes[1,1].set_xlabel("Residual ($/MWh)")
    axes[1,1].legend()
    axes[1,1].set_xlim(-200, 200)
    
    plt.tight_layout()
    out3_path = os.path.join(fig_dir, "Fig_Distribution_Shift.png")
    plt.savefig(out3_path, dpi=config.PLOT_DPI)
    plt.close()
    print(f"Saved -> {out3_path}")
    print("Done!")

if __name__ == "__main__":
    main()
