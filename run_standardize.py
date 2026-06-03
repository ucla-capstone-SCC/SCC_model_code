import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\emily\Downloads\TS_10_imputed.csv')

id_cols   = ['zip_code', 'year']
data_cols = [c for c in df.columns if c not in id_cols]

df_std = df[id_cols].copy()

means = df[data_cols].mean()
stds  = df[data_cols].std(ddof=1)   # sample std, same as sklearn StandardScaler default

df_std[data_cols] = (df[data_cols] - means) / stds

# Verify
print(f"Shape: {df_std.shape}")
print(f"\nPost-standardization check (should be ~0 mean, ~1 std):")
print(f"  Max |mean|: {df_std[data_cols].mean().abs().max():.2e}")
print(f"  Std range:  {df_std[data_cols].std().min():.4f} – {df_std[data_cols].std().max():.4f}")
print(f"\nMissing values: {df_std.isnull().sum().sum()}")

out_path = r'C:\Users\emily\Downloads\TS_10_imputed_standardized.csv'
df_std.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

# Save the scale parameters so you can invert later if needed
scale_df = pd.DataFrame({'variable': data_cols, 'mean': means.values, 'std': stds.values})
scale_path = r'C:\Users\emily\Downloads\TS_10_standardization_params.csv'
scale_df.to_csv(scale_path, index=False)
print(f"Scale params saved: {scale_path}")
