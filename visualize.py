"""
Visualizations for all experiments:
1. KF: True vs Estimated vs Noisy + Error over time
2. EKF: Pendulum angle tracking
3. Sensor Fusion: Single vs Dual sensor KF
4. Real Data: Raw accelerometer vs KF smoothed
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from simulate import run_experiment
from ekf import ExtendedKalmanFilter
from sensor_fusion import run_fusion_experiment
from real_data import run_real_data_experiment

COLORS = {
    "true":     "#2ecc71",
    "kf":       "#3498db",
    "noisy":    "#e74c3c",
    "fusion":   "#9b59b6",
    "raw":      "#e67e22",
    "error":    "#e74c3c",
    "bg":       "#0f1117",
    "grid":     "#2a2a3e",
    "text":     "#ecf0f1",
}

plt.rcParams.update({
    "figure.facecolor":  COLORS["bg"],
    "axes.facecolor":    COLORS["bg"],
    "axes.edgecolor":    COLORS["grid"],
    "axes.labelcolor":   COLORS["text"],
    "xtick.color":       COLORS["text"],
    "ytick.color":       COLORS["text"],
    "text.color":        COLORS["text"],
    "grid.color":        COLORS["grid"],
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "monospace",
    "legend.facecolor":  "#1a1a2e",
    "legend.edgecolor":  COLORS["grid"],
})


# ─────────────────────────────────────────────
# 1. KALMAN FILTER — Position tracking
# ─────────────────────────────────────────────
def plot_kf(ax_top, ax_err, result):
    t = np.arange(len(result["true_positions"]))

    ax_top.plot(t, result["observations"],       color=COLORS["noisy"],  alpha=0.4, lw=1,   label="Noisy Sensor")
    ax_top.plot(t, result["true_positions"],     color=COLORS["true"],   lw=1.8,            label="True Position")
    ax_top.plot(t, result["estimated_positions"],color=COLORS["kf"],     lw=2,              label="KF Estimate")
    ax_top.set_title("Kalman Filter — Position Tracking", fontsize=11, pad=8)
    ax_top.set_ylabel("Position")
    ax_top.legend(fontsize=8)
    ax_top.grid(True)

    err_kf  = np.abs(result["true_positions"] - result["estimated_positions"])
    err_raw = np.abs(result["true_positions"] - result["observations"])
    ax_err.fill_between(t, err_raw, alpha=0.25, color=COLORS["noisy"], label=f"Raw Error (RMSE={result['baseline_metrics']['RMSE']})")
    ax_err.plot(t, err_kf, color=COLORS["kf"], lw=1.5, label=f"KF Error  (RMSE={result['position_metrics']['RMSE']})")
    ax_err.set_title("Estimation Error Over Time", fontsize=11, pad=8)
    ax_err.set_ylabel("|Error|")
    ax_err.set_xlabel("Time Step")
    ax_err.legend(fontsize=8)
    ax_err.grid(True)


# ─────────────────────────────────────────────
# 2. EKF — Pendulum tracking
# ─────────────────────────────────────────────
def run_ekf_experiment(n_steps: int = 300):
    dt = 0.05
    g, L = 9.81, 1.0
    np.random.seed(7)

    ekf = ExtendedKalmanFilter(dt=dt, g=g, L=L, process_noise=0.005, measurement_noise=0.05)

    true_theta = [0.5]
    true_omega = [0.0]
    obs, est_theta = [], []

    theta, omega = 0.5, 0.0
    for _ in range(n_steps):
        omega += -(g / L) * np.sin(theta) * dt
        theta += omega * dt
        true_theta.append(theta)
        true_omega.append(omega)
        z = theta + np.random.normal(0, 0.22)
        obs.append(z)
        state = ekf.step(z)
        est_theta.append(state[0, 0])

    return np.array(true_theta[1:]), np.array(obs), np.array(est_theta)


def plot_ekf(ax, true_theta, obs, est_theta):
    t = np.arange(len(true_theta))
    ax.plot(t, obs,        color=COLORS["noisy"],  alpha=0.4, lw=1,   label="Noisy Observation")
    ax.plot(t, true_theta, color=COLORS["true"],   lw=1.8,            label="True Angle")
    ax.plot(t, est_theta,  color=COLORS["fusion"], lw=2,              label="EKF Estimate")
    rmse = np.sqrt(np.mean((true_theta - est_theta) ** 2))
    ax.set_title(f"Extended Kalman Filter — Pendulum (Non-linear)  |  RMSE={rmse:.4f}", fontsize=11, pad=8)
    ax.set_ylabel("Angle (rad)")
    ax.set_xlabel("Time Step")
    ax.legend(fontsize=8)
    ax.grid(True)


# ─────────────────────────────────────────────
# 3. SENSOR FUSION
# ─────────────────────────────────────────────
def plot_fusion(ax, result):
    t = np.arange(len(result["true_positions"]))
    ax.plot(t, result["obs_s1"],        color=COLORS["noisy"],  alpha=0.35, lw=1,   label=f"GPS Sensor  (RMSE={result['rmse_raw']:.3f})")
    ax.plot(t, result["true_positions"],color=COLORS["true"],   lw=1.8,            label="True Position")
    ax.plot(t, result["est_single"],    color=COLORS["kf"],     lw=1.8, ls="--",   label=f"Single KF   (RMSE={result['rmse_single']:.3f})")
    ax.plot(t, result["est_fusion"],    color=COLORS["fusion"], lw=2,              label=f"Fusion KF   (RMSE={result['rmse_fusion']:.3f})")
    ax.set_title("Sensor Fusion — GPS + Accelerometer", fontsize=11, pad=8)
    ax.set_ylabel("Position")
    ax.set_xlabel("Time Step")
    ax.legend(fontsize=8)
    ax.grid(True)


# ─────────────────────────────────────────────
# 4. REAL DATA
# ─────────────────────────────────────────────
def plot_real(ax, result):
    t = np.arange(len(result["raw_signal"]))
    ax.plot(t, result["raw_signal"],    color=COLORS["raw"],  alpha=0.6, lw=1,   label="Raw Accelerometer")
    ax.plot(t, result["smoothed_signal"], color=COLORS["kf"], lw=2,              label="KF Smoothed")
    m = result["metrics"]
    title = f"{result['source']}\nNoise Reduction: {m['Noise Reduction (%)']}%  |  Residual RMSE: {m['Residual RMSE']}"
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_ylabel("Acceleration (g)")
    ax.set_xlabel("Sample")
    ax.legend(fontsize=8)
    ax.grid(True)


# ─────────────────────────────────────────────
# MAIN — assemble all plots
# ─────────────────────────────────────────────
def main():
    print("Running experiments...")
    kf_result     = run_experiment(process_noise=0.1, measurement_noise=4.0)
    ekf_theta, ekf_obs, ekf_est = run_ekf_experiment()
    fusion_result = run_fusion_experiment()
    real_result   = run_real_data_experiment()
    print("Generating plots...")

    fig = plt.figure(figsize=(18, 16), facecolor=COLORS["bg"])
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.3)

    ax_kf_top = fig.add_subplot(gs[0, 0])
    ax_kf_err = fig.add_subplot(gs[0, 1])
    ax_ekf    = fig.add_subplot(gs[1, 0])
    ax_fusion = fig.add_subplot(gs[1, 1])
    ax_real   = fig.add_subplot(gs[2, :])

    plot_kf(ax_kf_top, ax_kf_err, kf_result)
    plot_ekf(ax_ekf, ekf_theta, ekf_obs, ekf_est)
    plot_fusion(ax_fusion, fusion_result)
    plot_real(ax_real, real_result)

    fig.suptitle("Kalman Filter Suite — State Estimation & Sensor Fusion",
                 fontsize=14, fontweight="bold", y=0.995, color=COLORS["text"])

    plt.savefig("results.png", dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    print("Saved: results.png")
    plt.close()


if __name__ == "__main__":
    main()
