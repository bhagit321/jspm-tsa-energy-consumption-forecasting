"""
========================================================================
Energy Consumption Forecasting: ARIMA vs SARIMA
========================================================================
Dataset: Datetime + Load (hourly electricity demand)
Approach: Classical decomposition, ARIMA(2,1,0), SARIMA(2,1,0)(1,1,0,7)
Author: MTech Project
========================================================================
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── OPTIONAL: install statsmodels for production use ──────────────────────────
# pip install statsmodels
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.stattools import adfuller, acf, pacf
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# ─────────────────────────────────────────────────────────────────────────────

# =============================================================================
# 1. DATA LOADING & PREPROCESSING
# =============================================================================
def load_and_preprocess(filepath):
    """Load energy dataset with Datetime and Load columns."""
    df = pd.read_csv(filepath, parse_dates=['Datetime'])
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)

    # Handle missing values via forward fill then backward fill
    df['Load'] = df['Load'].ffill().bfill()

    # Aggregate hourly → daily (mean)
    daily = df.resample('D').mean()
    daily.columns = ['Load']

    print(f"Dataset: {len(df)} hourly records → {len(daily)} daily records")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Load stats: min={df['Load'].min():.1f}, max={df['Load'].max():.1f}, "
          f"mean={df['Load'].mean():.1f} MW")
    return df, daily


# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================================
def plot_time_series_overview(daily):
    """Plot raw series with rolling average."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(daily.index, daily['Load'], color='#2563EB', linewidth=0.8, alpha=0.7, label='Daily Load')
    roll = daily['Load'].rolling(30, center=True).mean()
    ax.plot(daily.index, roll, color='#DC2626', linewidth=2, label='30-day MA')
    ax.set_title('Energy Demand Time Series', fontweight='bold')
    ax.set_ylabel('Load (MW)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('ts_overview.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# 3. CLASSICAL DECOMPOSITION (Additive)
# =============================================================================
def classical_decompose(series, period=7):
    """
    Additive decomposition: Load = Trend + Seasonal + Residual
    Uses centered moving average for trend extraction.
    """
    # Trend: centered moving average
    trend = series.rolling(window=period, center=True).mean()

    # Detrended series
    detrended = series - trend

    # Seasonal component: average value at each period position
    seasonal_vals = []
    for i in range(period):
        mask = np.arange(len(detrended)) % period == i
        seasonal_vals.append(np.nanmean(detrended.values[mask]))
    seasonal_vals = np.array(seasonal_vals)
    seasonal_vals -= seasonal_vals.mean()  # Centre around zero

    seasonal = pd.Series(
        np.tile(seasonal_vals, len(series) // period + 1)[:len(series)],
        index=series.index
    )

    # Residual
    residual = series - trend - seasonal

    return trend, seasonal, residual


def plot_decomposition(series, period=7):
    """Plot the 4-panel decomposition."""
    trend, seasonal, residual = classical_decompose(series, period)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    components = [
        (series, 'Original Series', '#1E293B'),
        (trend, 'Trend Component', '#DC2626'),
        (seasonal, 'Seasonal Component (weekly)', '#16A34A'),
        (residual, 'Residual Component', '#7C3AED'),
    ]
    for ax, (comp, title, color) in zip(axes, components):
        ax.plot(comp, color=color, linewidth=0.9)
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('decomposition.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# 4. STATIONARITY ANALYSIS
# =============================================================================
def compute_acf(series, nlags=40):
    """Manual autocorrelation function."""
    series = series.dropna().values
    n = len(series)
    mean = series.mean()
    c0 = np.sum((series - mean)**2) / n
    return np.array([
        np.sum((series[:n-k] - mean) * (series[k:] - mean)) / (n * c0)
        for k in range(nlags + 1)
    ])


def compute_pacf(series, nlags=20):
    """Manual partial autocorrelation via Durbin-Levinson."""
    acf = compute_acf(series, nlags=nlags)
    pacf = np.zeros(nlags + 1)
    pacf[0] = 1.0
    for k in range(1, nlags + 1):
        R = np.array([[acf[abs(i-j)] for j in range(k)] for i in range(k)])
        rhs = acf[1:k+1]
        try:
            sol = np.linalg.solve(R, rhs)
            pacf[k] = sol[-1]
        except np.linalg.LinAlgError:
            pacf[k] = 0
    return pacf


def difference(series, d=1):
    """Apply d-th order differencing."""
    result = series.copy()
    for _ in range(d):
        result = result.diff().dropna()
    return result


def check_stationarity(series, alpha=0.05):
    """
    Manual ADF-like check using variance comparison after differencing.
    For production: use statsmodels adfuller().
    """
    diff1 = difference(series, d=1)
    var_ratio = diff1.var() / series.var()
    print(f"Original variance: {series.var():.2f}")
    print(f"Differenced variance: {diff1.var():.2f}")
    print(f"Variance ratio (diff/orig): {var_ratio:.4f}")
    if var_ratio < 0.5:
        print("✓ Likely stationary after d=1 differencing")
    else:
        print("⚠ May require d=2 or seasonal differencing")
    return diff1


# =============================================================================
# 5. AR MODEL (Yule-Walker Estimation)
# =============================================================================
def fit_ar_yw(series, p):
    """
    Fit AR(p) model using Yule-Walker equations.
    Returns: (phi coefficients, series mean, noise variance)
    """
    series = series.dropna().values
    n = len(series)
    mean = series.mean()
    s = series - mean

    # Autocorrelation vector
    r = np.array([np.dot(s[:n-k], s[k:]) / n for k in range(p+1)])

    # Toeplitz autocorrelation matrix
    R = np.array([[r[abs(i-j)] for j in range(p)] for i in range(p)])
    r_vec = r[1:p+1]

    try:
        phi = np.linalg.solve(R, r_vec)
    except np.linalg.LinAlgError:
        phi = np.zeros(p)

    sigma2 = r[0] - np.dot(phi, r_vec)
    return phi, mean, max(sigma2, 0)


# =============================================================================
# 6. ARIMA FORECASTING  [ARIMA(p, d, q)]
# =============================================================================
def arima_forecast(train_series, p=2, d=1, steps=30):
    """
    ARIMA(p, d, q=0) forecast.
    Differences the series d times, fits AR(p), then integrates back.

    Parameters
    ----------
    train_series : pd.Series  - training data
    p            : int        - autoregressive order
    d            : int        - degree of differencing
    steps        : int        - forecast horizon

    Returns
    -------
    np.ndarray - forecasted values in original scale
    """
    # Step 1: Difference the series
    diff_series = difference(train_series, d=d)

    # Step 2: Fit AR(p) on differenced series
    phi, mu, sigma2 = fit_ar_yw(diff_series, p)
    print(f"ARIMA({p},{d},0) | AR coefficients: {phi.round(4)}")

    # Step 3: Forecast differenced values
    history = list(diff_series.values)
    forecasts_diff = []
    for _ in range(steps):
        val = mu + sum(phi[i] * (history[-(i+1)] - mu) for i in range(p))
        forecasts_diff.append(val)
        history.append(val)

    # Step 4: Integrate (undo differencing) d times
    forecasts = np.array(forecasts_diff)
    last_vals = train_series.values[-(d):]

    # For d=1: cumsum starting from last observed value
    result = np.zeros(steps)
    base = train_series.iloc[-1]
    for i in range(steps):
        base += forecasts_diff[i]
        result[i] = base

    return result


# =============================================================================
# 7. SARIMA FORECASTING  [SARIMA(p,d,q)(P,D,Q,m)]
# =============================================================================
def sarima_forecast(train_series, p=2, d=1, P=1, D=1, m=7, steps=30):
    """
    SARIMA(p,d,q=0)(P,D,Q=0,m) forecast.
    Applies seasonal + regular differencing, fits AR, then reconstructs.

    Parameters
    ----------
    train_series : pd.Series  - training data
    p, d         : ARIMA order
    P, D         : seasonal AR order, seasonal differencing degree
    m            : seasonal period (7=weekly, 12=monthly, 24=daily)
    steps        : forecast horizon

    Returns
    -------
    np.ndarray - forecasted values in original scale
    """
    # Step 1: Regular differencing (d times)
    diff_reg = difference(train_series, d=d)

    # Step 2: Seasonal differencing (D times with lag m)
    diff_seasonal = diff_reg.copy()
    for _ in range(D):
        diff_seasonal = diff_seasonal.diff(m).dropna()

    # Step 3: Fit AR(p + P*m) on double-differenced series
    effective_p = p + P * m
    phi, mu, sigma2 = fit_ar_yw(diff_seasonal, effective_p)
    print(f"SARIMA({p},{d},0)({P},{D},0,{m}) | AR order used: {effective_p}")

    # Step 4: Generate forecasts in double-differenced space
    history = list(diff_seasonal.values)
    forecasts_dd = []
    for _ in range(steps):
        val = mu + sum(phi[i] * (history[-(i+1)] - mu) for i in range(effective_p))
        forecasts_dd.append(val)
        history.append(val)

    # Step 5: Reconstruct — undo seasonal differencing (add back lag-m values)
    diff1_history = list(diff_reg.values[-m*3:])
    for dd_val in forecasts_dd:
        new_d1 = dd_val + diff1_history[-m]
        diff1_history.append(new_d1)
    new_diff1 = diff1_history[m*3:]

    # Step 6: Reconstruct — undo regular differencing
    train_history = list(train_series.values[-m*3:])
    result = []
    for d1_val in new_diff1[:steps]:
        new_val = d1_val + train_history[-1]
        result.append(new_val)
        train_history.append(new_val)

    return np.array(result[:steps])


# =============================================================================
# 8. EVALUATION METRICS
# =============================================================================
def evaluate_model(actual, predicted, model_name):
    """Compute RMSE, MAE, MAPE and print results."""
    actual, predicted = np.array(actual), np.array(predicted)
    rmse = np.sqrt(np.mean((actual - predicted)**2))
    mae  = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    print(f"\n{model_name} Performance:")
    print(f"  RMSE : {rmse:.2f} MW")
    print(f"  MAE  : {mae:.2f} MW")
    print(f"  MAPE : {mape:.2f}%")
    return {'Model': model_name, 'RMSE': rmse, 'MAE': mae, 'MAPE (%)': mape}


# =============================================================================
# 9. MAIN PIPELINE
# =============================================================================
if __name__ == '__main__':

    # ── Load data
    # Replace 'energy_data.csv' with your actual file path
    df, daily = load_and_preprocess('../data/energy_dataset.csv')

    # ── Decompose
    trend, seasonal, residual = classical_decompose(daily['Load'], period=7)
    print("\n[Decomposition] Components extracted successfully.")

    # ── Stationarity check
    print("\n[Stationarity Analysis]")
    diff1 = check_stationarity(daily['Load'])

    # ── Train/Test split (last 30 days as test)
    FORECAST_HORIZON = 30
    train_size = len(daily) - FORECAST_HORIZON
    train = daily['Load'][:train_size]
    test  = daily['Load'][train_size:]

    # ── ARIMA Forecast
    print("\n[ARIMA Forecasting]")
    arima_pred = arima_forecast(train, p=2, d=1, steps=FORECAST_HORIZON)
    arima_metrics = evaluate_model(test.values[:FORECAST_HORIZON], arima_pred, 'ARIMA(2,1,0)')

    # ── SARIMA Forecast
    print("\n[SARIMA Forecasting]")
    sarima_pred = sarima_forecast(train, p=2, d=1, P=1, D=1, m=7, steps=FORECAST_HORIZON)
    sarima_metrics = evaluate_model(test.values[:FORECAST_HORIZON], sarima_pred,
                                    'SARIMA(2,1,0)(1,1,0,7)')

    # ── Comparison
    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)
    comparison_df = pd.DataFrame([arima_metrics, sarima_metrics])
    comparison_df['RMSE'] = comparison_df['RMSE'].round(2)
    comparison_df['MAE']  = comparison_df['MAE'].round(2)
    comparison_df['MAPE (%)'] = comparison_df['MAPE (%)'].round(2)
    print(comparison_df.to_string(index=False))

    rmse_improvement = (arima_metrics['RMSE'] - sarima_metrics['RMSE']) / arima_metrics['RMSE'] * 100
    print(f"\nSARIMA improves RMSE by {rmse_improvement:.1f}% over ARIMA")
    print("\n✅ Recommendation: Use SARIMA for energy demand with weekly seasonality.")
    print("   ARIMA is suitable only when seasonal patterns are absent or period > 30.")

    # ── Seasonal insights
    print("\n[Energy Management Insights]")
    hourly_profile = df.groupby(df.index.hour)['Load'].mean()
    peak_hour = hourly_profile.idxmax()
    trough_hour = hourly_profile.idxmin()
    print(f"  Peak hour: {peak_hour:02d}:00 ({hourly_profile[peak_hour]:.0f} MW)")
    print(f"  Off-peak hour: {trough_hour:02d}:00 ({hourly_profile[trough_hour]:.0f} MW)")
    dow_profile = df.groupby(df.index.dayofweek)['Load'].mean()
    days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    peak_day = days[dow_profile.idxmax()]
    print(f"  Peak day of week: {peak_day} ({dow_profile.max():.0f} MW)")
    print(f"  Weekday vs weekend diff: {(dow_profile[:5].mean() - dow_profile[5:].mean()):.0f} MW")
