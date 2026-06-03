"""
Compose QoL intermediate indices and composite index from
arimax_forecast_TS10_standardized_long.csv

Steps per variable group:
  1. Per-year min-max scale each input variable to 0-100
     (data is already direction-adjusted: higher z-score = better)
  2. Equal-weighted average within each intermediate index
  3. Composite = 0.25 * economic + 0.25 * housing_transport_env
                + 0.25 * healthcare + 0.25 * food

Input variables used (then dropped from output):
  Economic (3):             pct_adults_60_plus_under_200_fpl,
                            ACS_MEDIAN_HH_INC_ZC, ACS_PCT_INC50_ABOVE65_ZC
  Housing/Transport/Env (2):ACS_PCT_RENTER_HU_ABOVE65_ZC,
                            ACS_PCT_HH_ALONE_ABOVE65_ZC
  Healthcare (4):           uninsured_65_plus, POS_DIST_MEDSURG_ICU_ZP,
                            POS_DIST_CLINIC_ZP, CCBP_TOT_HOME_ZP
  Food (2):                 CCBP_TOT_SOGS_ZP, CCBP_TOT_CFS_ZP

Remaining 21 variables kept as-is (standardized z-scores).
Output: arimax_forecast_TS10_qol_indices.csv
"""
import pandas as pd
import numpy as np

INPUT  = r"C:\Users\emily\Downloads\arimax_forecast_TS10_standardized_long.csv"
OUTPUT = r"C:\Users\emily\Downloads\arimax_forecast_TS10_qol_indices.csv"

INDEX_VARS = {
    'qol_economic_intermediate_index': [
        'pct_adults_60_plus_under_200_fpl',
        'ACS_MEDIAN_HH_INC_ZC',
        'ACS_PCT_INC50_ABOVE65_ZC',
    ],
    'qol_housing_transport_environment_intermediate_index': [
        'ACS_PCT_RENTER_HU_ABOVE65_ZC',
        'ACS_PCT_HH_ALONE_ABOVE65_ZC',
    ],
    'qol_healthcare_intermediate_index': [
        'uninsured_65_plus',
        'POS_DIST_MEDSURG_ICU_ZP',
        'POS_DIST_CLINIC_ZP',
        'CCBP_TOT_HOME_ZP',
    ],
    'qol_food_intermediate_index': [
        'CCBP_TOT_SOGS_ZP',
        'CCBP_TOT_CFS_ZP',
    ],
}
ALL_INDEX_VARS = [v for vlist in INDEX_VARS.values() for v in vlist]

df = pd.read_csv(INPUT)

# ── Per-year min-max scaling for each index input variable ────────────────────
scaled = df.copy()
for yr in sorted(df['year'].unique()):
    mask = df['year'] == yr
    for var in ALL_INDEX_VARS:
        vals = df.loc[mask, var]
        mn, mx = vals.min(), vals.max()
        if mx > mn:
            scaled.loc[mask, var] = (vals - mn) / (mx - mn) * 100.0
        else:
            scaled.loc[mask, var] = 50.0  # all ZIPs identical for this var/year

# ── Compute intermediate indices (equal-weighted average) ─────────────────────
for idx_name, vars_ in INDEX_VARS.items():
    scaled[idx_name] = scaled[vars_].mean(axis=1)
    print(f"  {idx_name}: {len(vars_)} vars, "
          f"range [{scaled[idx_name].min():.2f}, {scaled[idx_name].max():.2f}]")

# ── Composite QoL index ───────────────────────────────────────────────────────
intermediate_cols = list(INDEX_VARS.keys())
scaled['qol_composite_index'] = scaled[intermediate_cols].mean(axis=1)

# ── Drop the 11 input variables ───────────────────────────────────────────────
result = scaled.drop(columns=ALL_INDEX_VARS)

# Reorder: zip_code, year, composite, intermediates, then remaining vars
remaining_vars = [c for c in result.columns
                  if c not in ['zip_code', 'year', 'qol_composite_index'] + intermediate_cols]
col_order = (['zip_code', 'year', 'qol_composite_index'] +
             intermediate_cols + remaining_vars)
result = result[col_order].sort_values(['zip_code', 'year']).reset_index(drop=True)

result.to_csv(OUTPUT, index=False)

print(f"\nDone. Saved: {OUTPUT}")
print(f"  Rows   : {len(result)}")
print(f"  Columns: {len(result.columns)}")
print(f"\n  Dropped (used in indices): {len(ALL_INDEX_VARS)} vars")
print(f"  Kept as-is (z-scored)    : {len(remaining_vars)} vars")
print(f"  Added index columns      : {len(intermediate_cols)} intermediate + 1 composite")

print(f"\nQoL composite index summary (all years):")
print(result['qol_composite_index'].describe().round(2).to_string())

print(f"\nSample output (first 3 rows):")
show_cols = ['zip_code', 'year', 'qol_composite_index'] + intermediate_cols
print(result[show_cols].head(3).to_string())
