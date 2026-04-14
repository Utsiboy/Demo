"""
Alpha Signal Evaluation Pipeline

This script demonstrates a simple workflow for:
1. Generating a microstructure-inspired signal
2. Computing forward returns (markouts)
3. Evaluating signal quality using quantile bucketing
4. Computing basic statistical significance

Author: Utsav Sen
"""

import pandas as pd
import numpy as np


def generate_data(n=5000):
    """Simulate price series resembling high-frequency data"""
    np.random.seed(42)

    timestamps = pd.date_range(start="2023-01-01", periods=n, freq="S")
    price = 100 + np.cumsum(np.random.randn(n) * 0.02)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'price': price
    })

    return df


def compute_signal(df):
    """
    Generate a normalized short-term momentum signal
    (proxy for microstructure alpha)
    """
    df['ret'] = df['price'].pct_change()

    rolling_mean = df['ret'].rolling(20).mean()
    rolling_std = df['ret'].rolling(20).std()

    df['signal'] = rolling_mean / (rolling_std + 1e-6)

    return df


def compute_markouts(df, horizon=10):
    """
    Compute forward returns (markouts)
    """
    df['future_ret'] = df['price'].pct_change(horizon).shift(-horizon)
    return df


def quantile_analysis(df, n=5):
    """
    Bucket signal into quantiles and compute:
    - mean return
    - std deviation
    - count
    - t-statistic
    """
    df = df.dropna().copy()

    df['bucket'] = pd.qcut(df['signal'], n, labels=False)

    stats = df.groupby('bucket')['future_ret'].agg(['mean', 'std', 'count'])

    stats['t_stat'] = stats['mean'] / (stats['std'] / np.sqrt(stats['count']))

    return stats


def main():
    df = generate_data()
    df = compute_signal(df)
    df = compute_markouts(df)

    results = quantile_analysis(df)

    print("\nQuantile Analysis Results:")
    print(results)

    print("\nInterpretation:")
    print("Monotonic increase/decrease across buckets indicates a useful signal.")


if __name__ == "__main__":
    main()
