import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.structural import UnobservedComponents

# ── Load and prep ──────────────────────────────────────────────────────────────
df = pd.read_csv(r'C:\Users\emily\Downloads\TS_10_selected_variables.csv')

# Step 0: Sort by ZIP then year
df = df.sort_values(['zip_code', 'year']).reset_index(drop=True)
print(f"Step 0: sorted by ZIP and year -> {len(df)} rows, {df['zip_code'].nunique()} ZIPs")

vars_only = [c for c in df.columns if c not in ['zip_code', 'year']]

# Compute missingness on working dataset
miss    = df[vars_only].isnull().sum()
miss_p  = (miss / len(df) * 100).round(2)
before_miss = miss.copy()

# ── Tier classification ────────────────────────────────────────────────────────
TIER_MAP = {}
for v in vars_only:
    p = miss_p[v]
    if p == 0:
        t = 'A'
    elif v in ('pct_no_computer', 'pct_no_internet'):
        t = 'C'
    elif p <= 5:
        t = 'B'
    elif p <= 35:
        t = 'D'
    elif p <= 60:
        t = 'E'
    else:
        t = 'F'
    TIER_MAP[v] = t

CCBP_VARS = [v for v in vars_only if v.startswith('CCBP')]
methods_used = {}

# ── Kalman smoother helper ─────────────────────────────────────────────────────
def kalman_smooth(series, min_obs=3):
    """Return series with NaNs filled via Kalman smoother, or None if insufficient data."""
    if series.notna().sum() < min_obs:
        return None
    y = series.values.astype(float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for spec in ['local linear trend', 'local level']:
                try:
                    mod = UnobservedComponents(y, spec)
                    res = mod.fit(disp=False, method='bfgs')
                    smoothed = res.smoothed_state[0]
                    filled = series.copy().astype(float)
                    filled[series.isna()] = smoothed[series.isna().values]
                    return filled
                except Exception:
                    continue
    except Exception:
        pass
    return None

# ── Shared Kalman-or-county-median imputer for Tiers D/E/F ────────────────────
def impute_kalman_or_median(df, v, tier_label):
    """Per-ZIP: Kalman if >=3 obs, else scale county-median time series."""
    # County median per year; fill NaN years with overall median
    county_ts = df.groupby('year')[v].median()
    overall_med = df[v].median()
    if np.isnan(overall_med):
        overall_med = 0.0
    county_ts = county_ts.fillna(overall_med)

    zip_methods = {}
    for zip_code in df['zip_code'].unique():
        mask = df['zip_code'] == zip_code
        s = df.loc[mask, v].copy()
        if s.isnull().sum() == 0:
            continue

        n_obs = s.notna().sum()

        if n_obs >= 3:
            result = kalman_smooth(s, min_obs=3)
            if result is not None:
                df.loc[mask, v] = result
                zip_methods[zip_code] = 'kalman'
                continue

        # Fallback: county-median scaling
        if n_obs > 0:
            obs_df = df.loc[mask & df[v].notna(), ['year', v]]
            ratios = []
            for _, row in obs_df.iterrows():
                yr = int(row['year'])
                cm = county_ts[yr] if yr in county_ts.index else overall_med
                if cm > 0:
                    ratios.append(row[v] / cm)
            scale = float(np.nanmean(ratios)) if ratios else 1.0
        else:
            scale = 1.0

        for idx in df.loc[mask & df[v].isnull()].index:
            yr = int(df.loc[idx, 'year'])
            cm = county_ts[yr] if yr in county_ts.index else overall_med
            df.loc[idx, v] = max(0.0, cm * scale)

        zip_methods[zip_code] = f'county_median_scaled(scale={scale:.2f})'

    n_kalman = sum(1 for m in zip_methods.values() if m == 'kalman')
    n_median = len(zip_methods) - n_kalman
    print(f"  {v:<50} kalman={n_kalman} ZIPs  median_scaled={n_median} ZIPs")
    return zip_methods


# ══════════════════════════════════════════════════════════════════════════════
# TIER B — Within-ZIP linear interpolation + forward/backward fill
# ══════════════════════════════════════════════════════════════════════════════
tier_b_vars = [v for v, t in TIER_MAP.items() if t == 'B']
print(f"\n--- Tier B ({len(tier_b_vars)} vars): linear interpolation + ffill/bfill ---")
for v in tier_b_vars:
    for zip_code in df['zip_code'].unique():
        mask = df['zip_code'] == zip_code
        s = df.loc[mask, v]
        df.loc[mask, v] = s.interpolate(method='linear').ffill().bfill()
    methods_used[v] = 'Tier B: linear interpolation + ffill/bfill'
    print(f"  {v}")


# ══════════════════════════════════════════════════════════════════════════════
# TIER C — Backcast 2015-2016 from 2017-2019 OLS slope, then interpolate remainder
# ══════════════════════════════════════════════════════════════════════════════
tier_c_vars = [v for v, t in TIER_MAP.items() if t == 'C']
print(f"\n--- Tier C ({len(tier_c_vars)} vars): linear backcast 2015-2016 ---")
for v in tier_c_vars:
    for zip_code in df['zip_code'].unique():
        mask = df['zip_code'] == zip_code
        zip_df = df.loc[mask]

        ref = zip_df[zip_df['year'].isin([2017, 2018, 2019])][['year', v]].dropna()

        if len(ref) >= 2:
            coeffs = np.polyfit(ref['year'].values.astype(float), ref[v].values.astype(float), 1)
            for yr in [2015, 2016]:
                idx_rows = zip_df[zip_df['year'] == yr].index
                if len(idx_rows) and pd.isnull(df.loc[idx_rows[0], v]):
                    df.loc[idx_rows[0], v] = max(0.0, float(np.polyval(coeffs, yr)))
        elif len(ref) == 1:
            val = float(ref[v].values[0])
            for yr in [2015, 2016]:
                idx_rows = zip_df[zip_df['year'] == yr].index
                if len(idx_rows) and pd.isnull(df.loc[idx_rows[0], v]):
                    df.loc[idx_rows[0], v] = max(0.0, val)
        else:
            # No 2017-2019 reference: use county median for those years
            for yr in [2015, 2016]:
                cm = df[df['year'] == yr][v].median()
                idx_rows = zip_df[zip_df['year'] == yr].index
                if len(idx_rows) and pd.isnull(df.loc[idx_rows[0], v]):
                    df.loc[idx_rows[0], v] = max(0.0, cm) if not np.isnan(cm) else 0.0

        # Mop up any sporadic remaining NaNs (e.g. 2019-2021 gaps)
        df.loc[mask, v] = df.loc[mask, v].interpolate(method='linear').ffill().bfill()

    methods_used[v] = 'Tier C: OLS backcast to 2015-2016 + interpolate remainder'
    print(f"  {v}")


# ══════════════════════════════════════════════════════════════════════════════
# TIER D — Kalman smoother (≥3 obs) or county-median scaling
# ══════════════════════════════════════════════════════════════════════════════
tier_d_vars = [v for v, t in TIER_MAP.items() if t == 'D']
print(f"\n--- Tier D ({len(tier_d_vars)} vars): Kalman / county-median ---")
for v in tier_d_vars:
    impute_kalman_or_median(df, v, 'D')
    methods_used[v] = 'Tier D: Kalman smoother / county-median scaling'


# ══════════════════════════════════════════════════════════════════════════════
# TIER E — same approach, flagged as high uncertainty
# ══════════════════════════════════════════════════════════════════════════════
tier_e_vars = [v for v, t in TIER_MAP.items() if t == 'E']
print(f"\n--- Tier E ({len(tier_e_vars)} vars): Kalman / county-median [HIGH UNCERTAINTY] ---")
for v in tier_e_vars:
    impute_kalman_or_median(df, v, 'E')
    methods_used[v] = 'Tier E: Kalman smoother / county-median scaling [HIGH UNCERTAINTY]'


# ══════════════════════════════════════════════════════════════════════════════
# TIER F — kept per user request; same Kalman / county-median approach
# ══════════════════════════════════════════════════════════════════════════════
tier_f_vars = [v for v, t in TIER_MAP.items() if t == 'F']
print(f"\n--- Tier F ({len(tier_f_vars)} vars): Kalman / county-median [EXTREME UNCERTAINTY — kept per request] ---")
for v in tier_f_vars:
    impute_kalman_or_median(df, v, 'F')
    methods_used[v] = 'Tier F: Kalman smoother / county-median scaling [EXTREME UNCERTAINTY]'


# ══════════════════════════════════════════════════════════════════════════════
# Round CCBP count variables to nearest non-negative integer
# ══════════════════════════════════════════════════════════════════════════════
for v in CCBP_VARS:
    df[v] = df[v].astype(float).clip(lower=0).round().astype(int)

# ══════════════════════════════════════════════════════════════════════════════
# Final verification
# ══════════════════════════════════════════════════════════════════════════════
after_miss = df[vars_only].isnull().sum()
total_before = int(before_miss.sum())
total_after  = int(after_miss.sum())

print(f"\n{'='*80}")
print(f"IMPUTATION COMPLETE")
print(f"Total missing before: {total_before:,}")
print(f"Total missing after:  {total_after:,}")
if total_after > 0:
    print("WARNING: some missing values remain!")
    print(after_miss[after_miss > 0])
print(f"{'='*80}")

print(f"\n{'Variable':<50} {'Tier':<6} {'Before':>8} {'After':>8}  Method")
print("-" * 115)
for v in vars_only:
    tier   = TIER_MAP.get(v, 'A')
    b      = int(before_miss[v])
    a      = int(after_miss[v])
    method = methods_used.get(v, 'None (Tier A — complete)')
    print(f"{v:<50} {tier:<6} {b:>8} {a:>8}  {method}")

# ══════════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════════
out_path = r'C:\Users\emily\Downloads\TS_10_imputed.csv'
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"Shape: {df.shape}  ({df['zip_code'].nunique()} ZIPs x {df['year'].nunique()} years)")
