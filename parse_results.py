import pandas as pd
import re

print("--- PJM Point Accuracy ---")
try:
    df_pjm = pd.read_csv('reports/table_point_accuracy_full_pjm.csv')
    print(df_pjm[['Model', 'MAE', 'RMSE']].to_string(index=False))
except Exception as e:
    print(f"Error reading PJM: {e}")

print("\n--- ERCOT Point Accuracy ---")
try:
    df_ercot = pd.read_csv('reports/table_point_accuracy_full_ercot.csv')
    print(df_ercot[['Model', 'MAE', 'RMSE']].to_string(index=False))
except Exception as e:
    print(f"Error reading ERCOT: {e}")

print("\n--- Rolling PatchTST (PJM) ---")
try:
    df_rp_pjm = pd.read_csv('reports/table_rolling_patchtst_pjm.csv')
    print(df_rp_pjm.head(3))
except:
    pass

print("\n--- Rolling PatchTST (ERCOT) ---")
try:
    df_rp_ercot = pd.read_csv('reports/table_rolling_patchtst_ercot.csv')
    print(df_rp_ercot.head(3))
except:
    pass

