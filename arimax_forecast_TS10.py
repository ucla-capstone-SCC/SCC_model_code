"""
ARIMAX Forecast for TS_10_imputed.csv
Train: 2015-2022  |  Test: 2023-2024  |  Forecast: 2025-2029
Model: ARIMAX(p,d,q) with pandemic as exogenous intervention variable.
       Order selected by AIC grid search (p,q in 0-2, d in 0-1).
       pandemic is NOT forecasted — fixed to 0 for 2025-2029.
Output: arimax_forecast_TS10_results.csv (wide format, full 2015-2029)
  Columns: zip_code, variable, model, aic, note,
           obs_2015..obs_2024,
           pred_2023, pred_2024  (test-period model predictions),
           forecast_2025..forecast_2029,
           ci_lower_2025..ci_upper_2029
"""
import warnings, itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
warnings.filterwarnings("ignore")

INPUT  = r"C:\Users\emily\Downloads\TS_10_imputed.csv"
OUTPUT = r"C:\Users\emily\Downloads\arimax_forecast_TS10_results.csv"

TRAIN_END      = 2022
TEST_YEARS     = [2023, 2024]
FORECAST_YEARS = [2025, 2026, 2027, 2028, 2029]

df = pd.read_csv(INPUT)
df = df.sort_values(["zip_code", "year"]).reset_index(drop=True)

# pandemic is exogenous — excluded from the list of target variables
VARIABLES = [c for c in df.columns if c not in ("zip_code", "year", "pandemic")]
ZIP_CODES = sorted(df["zip_code"].unique())

ORDERS = [(p, d, q) for p, d, q in itertools.product(range(3), range(2), range(3))
          if p + d + q > 0 and p + q <= 3]


def pick_arimax(series, exog):
    """Grid-search ARIMAX order by AIC; fallback to (0,1,0) on total failure."""
    best_aic, best_fit, best_ord = np.inf, None, (1, 1, 0)
    for ord_ in ORDERS:
        p, d, q = ord_
        if len(series) - d <= p + q + 2:
            continue
        try:
            fit = ARIMA(series, exog=exog, order=ord_,
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit(
                            method_kwargs={"warn_convergence": False})
            if fit.aic < best_aic:
                best_aic, best_fit, best_ord = fit.aic, fit, ord_
        except Exception:
            pass
    if best_fit is None:
        try:
            best_fit = ARIMA(series, exog=exog, order=(0, 1, 0)).fit()
            best_ord = (0, 1, 0)
        except Exception:
            pass
    return best_fit, best_ord


rows = []
total = len(ZIP_CODES) * len(VARIABLES)
done  = 0
print(f"Fitting {total} ARIMAX models ({len(ZIP_CODES)} ZIPs × {len(VARIABLES)} vars)...")

for zc in ZIP_CODES:
    sub = df[df["zip_code"] == zc].set_index("year")
    pandemic_exog = sub["pandemic"].astype(float).values.reshape(-1, 1)

    for var in VARIABLES:
        done += 1
        if done % 300 == 0:
            print(f"  {done}/{total}...")

        full_y = sub[var].astype(float).interpolate("linear").ffill().bfill()
        all_years = full_y.index.to_numpy()

        train_mask = all_years <= TRAIN_END
        test_mask  = np.isin(all_years, TEST_YEARS)

        # Keep as pandas Series so statsmodels returns Series from forecast()
        train_y    = full_y.iloc[train_mask]
        train_exog = pandemic_exog[train_mask]
        test_y     = full_y.iloc[test_mask]
        test_exog  = pandemic_exog[test_mask]

        # Base row: observed values 2015-2024
        row = {"zip_code": zc, "variable": var}
        for yr in range(2015, 2025):
            idx = np.where(all_years == yr)[0]
            row[f"obs_{yr}"] = round(float(full_y.values[idx[0]]), 6) if len(idx) else np.nan

        if (~np.isnan(train_y.values)).sum() < 4:
            for yr in TEST_YEARS:
                row[f"pred_{yr}"] = np.nan
            for yr in FORECAST_YEARS:
                row[f"forecast_{yr}"] = np.nan
                row[f"ci_lower_{yr}"] = np.nan
                row[f"ci_upper_{yr}"] = np.nan
            row.update({"model": "N/A", "aic": np.nan, "note": "insufficient data"})
            rows.append(row)
            continue

        fit_tr, order = pick_arimax(train_y, train_exog)

        if fit_tr is None:
            for yr in TEST_YEARS:
                row[f"pred_{yr}"] = np.nan
            for yr in FORECAST_YEARS:
                row[f"forecast_{yr}"] = np.nan
                row[f"ci_lower_{yr}"] = np.nan
                row[f"ci_upper_{yr}"] = np.nan
            row.update({"model": "N/A", "aic": np.nan, "note": "fit failed"})
            rows.append(row)
            continue

        # Test predictions (2023-2024) from train-only model
        try:
            test_pred = np.asarray(fit_tr.forecast(steps=len(TEST_YEARS), exog=test_exog))
            for i, yr in enumerate(TEST_YEARS):
                row[f"pred_{yr}"] = round(float(test_pred[i]), 6)
        except Exception:
            for yr in TEST_YEARS:
                row[f"pred_{yr}"] = np.nan

        # Refit on full 2015-2024, then forecast 2025-2029 with pandemic=0
        future_exog = np.zeros((5, 1))
        try:
            # Pass full_y as pandas Series so forecast() returns a Series
            fit_full = ARIMA(full_y, exog=pandemic_exog, order=order,
                             enforce_stationarity=False,
                             enforce_invertibility=False).fit(
                                 method_kwargs={"warn_convergence": False})
            fc    = fit_full.get_forecast(steps=5, exog=future_exog)
            fmean = np.asarray(fc.predicted_mean)
            fci   = np.asarray(fc.conf_int(alpha=0.05))
        except Exception:
            # Fallback: extend train-model forecast past test period
            try:
                fc    = fit_tr.get_forecast(steps=5 + len(TEST_YEARS),
                                            exog=np.vstack([test_exog, future_exog]))
                fmean = np.asarray(fc.predicted_mean)[len(TEST_YEARS):]
                fci   = np.asarray(fc.conf_int(alpha=0.05))[len(TEST_YEARS):]
            except Exception:
                fmean = [np.nan] * 5
                fci   = [[np.nan, np.nan]] * 5

        for i, yr in enumerate(FORECAST_YEARS):
            row[f"forecast_{yr}"] = (round(float(fmean[i]), 6)
                                     if i < len(fmean) and not np.isnan(fmean[i]) else np.nan)
            row[f"ci_lower_{yr}"] = (round(float(fci[i][0]), 6)
                                     if i < len(fci) and not np.isnan(fci[i][0]) else np.nan)
            row[f"ci_upper_{yr}"] = (round(float(fci[i][1]), 6)
                                     if i < len(fci) and not np.isnan(fci[i][1]) else np.nan)

        row.update({"model": f"ARIMAX{order}", "aic": round(float(fit_tr.aic), 4), "note": "ok"})
        rows.append(row)

print("Writing output CSV...")
result_df = pd.DataFrame(rows)

col_order = (
    ["zip_code", "variable", "model", "aic", "note"] +
    [f"obs_{yr}"      for yr in range(2015, 2025)] +
    [f"pred_{yr}"     for yr in TEST_YEARS] +
    [f"forecast_{yr}" for yr in FORECAST_YEARS] +
    [f"ci_lower_{yr}" for yr in FORECAST_YEARS] +
    [f"ci_upper_{yr}" for yr in FORECAST_YEARS]
)
result_df = result_df[[c for c in col_order if c in result_df.columns]]
result_df.to_csv(OUTPUT, index=False)

print(f"\nDone. Saved: {OUTPUT}")
print(f"  Rows    : {len(result_df)}  ({len(ZIP_CODES)} ZIPs × {len(VARIABLES)} variables)")
print(f"  Columns : {len(result_df.columns)}")

ok_pct = (result_df["note"] == "ok").mean() * 100
print(f"  Models OK: {ok_pct:.1f}%")

# Quick model-order summary
order_counts = result_df.loc[result_df["note"] == "ok", "model"].value_counts().head(5)
print("\nTop 5 most common ARIMAX orders:")
print(order_counts.to_string())
