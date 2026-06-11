"""
Simulate noisy sensor data and benchmark Kalman Filter estimation.
Metrics: RMSE, Log-Likelihood, MAE
"""

import numpy as np
import pandas as pd
from kalman_filter import KalmanFilter


def generate_time_series(n_steps: int = 200, dt: float = 1.0, seed: int = 42):
    """Simulate true position/velocity with Gaussian noise observations."""
    np.random.seed(seed)

    true_positions = []
    true_velocities = []
    noisy_observations = []

    pos, vel = 0.0, 1.0
    for _ in range(n_steps):
        vel += np.random.normal(0, 0.05)        # small acceleration noise
        pos += vel * dt
        obs = pos + np.random.normal(0, 2.0)    # sensor noise (std=2)

        true_positions.append(pos)
        true_velocities.append(vel)
        noisy_observations.append(obs)

    return np.array(true_positions), np.array(true_velocities), np.array(noisy_observations)


def compute_metrics(true_vals: np.ndarray, estimated_vals: np.ndarray, noise_std: float = 2.0):
    residuals = true_vals - estimated_vals
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    log_likelihood = -0.5 * np.sum((residuals / noise_std) ** 2 + np.log(2 * np.pi * noise_std ** 2))
    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "Log-Likelihood": round(log_likelihood, 4)}


def run_experiment(process_noise: float = 0.1, measurement_noise: float = 4.0, n_steps: int = 200):
    true_pos, true_vel, observations = generate_time_series(n_steps=n_steps)

    kf = KalmanFilter(dt=1.0, process_noise=process_noise, measurement_noise=measurement_noise)

    estimated_positions = []
    estimated_velocities = []

    for z in observations:
        state = kf.step(z)
        estimated_positions.append(state[0, 0])
        estimated_velocities.append(state[1, 0])

    estimated_positions = np.array(estimated_positions)
    estimated_velocities = np.array(estimated_velocities)

    pos_metrics = compute_metrics(true_pos, estimated_positions)
    vel_metrics = compute_metrics(true_vel, estimated_velocities)
    obs_metrics = compute_metrics(true_pos, observations)  # baseline: raw sensor

    return {
        "true_positions": true_pos,
        "true_velocities": true_vel,
        "observations": observations,
        "estimated_positions": estimated_positions,
        "estimated_velocities": estimated_velocities,
        "position_metrics": pos_metrics,
        "velocity_metrics": vel_metrics,
        "baseline_metrics": obs_metrics,
    }


def benchmark():
    print("=" * 55)
    print("  Kalman Filter State Estimator — Benchmark Report")
    print("=" * 55)

    configs = [
        {"process_noise": 0.05, "measurement_noise": 4.0,  "label": "Low Process Noise"},
        {"process_noise": 0.1,  "measurement_noise": 4.0,  "label": "Balanced"},
        {"process_noise": 0.5,  "measurement_noise": 4.0,  "label": "High Process Noise"},
        {"process_noise": 0.1,  "measurement_noise": 1.0,  "label": "Low Meas. Noise"},
        {"process_noise": 0.1,  "measurement_noise": 16.0, "label": "High Meas. Noise"},
    ]

    rows = []
    for cfg in configs:
        result = run_experiment(cfg["process_noise"], cfg["measurement_noise"])
        rows.append({
            "Config": cfg["label"],
            "Pos RMSE (KF)": result["position_metrics"]["RMSE"],
            "Pos RMSE (Raw)": result["baseline_metrics"]["RMSE"],
            "Pos MAE (KF)": result["position_metrics"]["MAE"],
            "Log-Likelihood": result["position_metrics"]["Log-Likelihood"],
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    best = df.loc[df["Pos RMSE (KF)"].idxmin()]
    worst_raw = df["Pos RMSE (Raw)"].max()
    improvement = round((1 - best["Pos RMSE (KF)"] / worst_raw) * 100, 1)

    print(f"\nBest config: '{best['Config']}'")
    print(f"KF RMSE: {best['Pos RMSE (KF)']}  |  Raw sensor RMSE: {best['Pos RMSE (Raw)']}")
    print(f"Estimation error reduction vs raw sensor: {improvement}%")
    print("=" * 55)

    return df


if __name__ == "__main__":
    benchmark()
