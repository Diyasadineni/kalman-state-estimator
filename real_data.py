"""
Real Dataset — Kalman Filter on real accelerometer time-series.
Uses UCI HAR (Human Activity Recognition) accelerometer data or
falls back to a structured synthetic dataset mimicking real sensor characteristics.

Dataset: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
We use the raw accelerometer signal from the Body Acceleration (total) axis.
"""

import numpy as np
import pandas as pd
import urllib.request
import os
import zipfile


DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
DATA_DIR = "data/uci_har"


def download_uci_har(data_dir: str = DATA_DIR) -> bool:
    """Download UCI HAR dataset. Returns True if successful."""
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "uci_har.zip")
    try:
        print("Downloading UCI HAR Dataset (~60MB)...")
        urllib.request.urlretrieve(DATASET_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(data_dir)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Download failed: {e}. Using fallback dataset.")
        return False


def load_real_accelerometer(data_dir: str = DATA_DIR) -> np.ndarray:
    """
    Load raw accelerometer signal from UCI HAR dataset.
    Returns 1D array of body acceleration (x-axis), first 500 samples.
    """
    accel_path = os.path.join(data_dir, "UCI HAR Dataset", "train", "Inertial Signals", "body_acc_x_train.txt")

    if not os.path.exists(accel_path):
        success = download_uci_har(data_dir)
        if not success or not os.path.exists(accel_path):
            return None

    data = pd.read_csv(accel_path, sep=r'\s+', header=None)
    # Each row is a window of 128 readings — flatten first few windows
    signal = data.iloc[:4].values.flatten()[:500]
    return signal.astype(np.float64)


def generate_realistic_fallback(n: int = 500, seed: int = 42) -> np.ndarray:
    """
    Structured fallback: simulates real accelerometer characteristics —
    low-freq motion signal + sensor noise + occasional spike artifacts.
    """
    np.random.seed(seed)
    t = np.linspace(0, 10, n)

    # Simulate walking-like motion (periodic + drift)
    signal = (
        0.3 * np.sin(2 * np.pi * 1.8 * t) +       # ~1.8 Hz walking cadence
        0.1 * np.sin(2 * np.pi * 3.6 * t) +        # harmonic
        0.05 * t / 10 +                              # slow drift
        np.random.normal(0, 0.08, n)                # sensor noise
    )
    # Occasional spike artifacts
    spike_idx = np.random.choice(n, size=8, replace=False)
    signal[spike_idx] += np.random.choice([-1, 1], size=8) * np.random.uniform(0.3, 0.6, size=8)

    return signal


def run_real_data_experiment():
    """Apply KF to real accelerometer signal; compare smoothed vs raw."""
    from kalman_filter import KalmanFilter

    print("Loading accelerometer data...")
    signal = load_real_accelerometer()

    if signal is None:
        print("Using realistic fallback dataset (walking motion simulation).")
        signal = generate_realistic_fallback()
        source = "Simulated Walking Accelerometer (fallback)"
    else:
        print(f"Loaded real UCI HAR accelerometer signal ({len(signal)} samples).")
        source = "UCI HAR Real Accelerometer (body_acc_x)"

    # Apply KF for signal smoothing / state estimation
    kf = KalmanFilter(dt=0.02, process_noise=0.001, measurement_noise=0.01)
    smoothed = []
    for z in signal:
        state = kf.step(z)
        smoothed.append(state[0, 0])

    smoothed = np.array(smoothed)
    residuals = signal - smoothed

    metrics = {
        "Signal Std (Raw)": round(float(np.std(signal)), 5),
        "Signal Std (Smoothed)": round(float(np.std(smoothed)), 5),
        "Residual RMSE": round(float(np.sqrt(np.mean(residuals ** 2))), 5),
        "Noise Reduction (%)": round((1 - np.std(smoothed) / np.std(signal)) * 100, 2),
    }

    return {
        "source": source,
        "raw_signal": signal,
        "smoothed_signal": smoothed,
        "metrics": metrics,
    }
