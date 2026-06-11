"""
Sensor Fusion — Kalman Filter with 2 noisy sensors.
Fuses GPS-like (low freq, high noise) and accelerometer-like (high freq, low noise) observations.
"""

import numpy as np


class SensorFusionKF:
    """
    Fuses two independent noisy sensors measuring position.

    Sensor 1 (GPS-like):      high noise, always available
    Sensor 2 (Accel-like):    low noise, available every N steps
    """

    def __init__(self, dt: float = 1.0, process_noise: float = 0.1,
                 noise_s1: float = 4.0, noise_s2: float = 0.5):
        self.dt = dt

        self.F = np.array([[1, dt], [0, 1]])
        self.Q = process_noise * np.array([[dt**4 / 4, dt**3 / 2],
                                            [dt**3 / 2, dt**2]])

        # Observation matrices for each sensor (both observe position)
        self.H1 = np.array([[1, 0]])
        self.H2 = np.array([[1, 0]])

        self.R1 = np.array([[noise_s1]])
        self.R2 = np.array([[noise_s2]])

        self.x = np.zeros((2, 1))
        self.P = np.eye(2) * 1000

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def _update(self, z, H, R):
        z = np.array([[z]])
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ H) @ self.P

    def step(self, z1: float, z2: float = None):
        """z1 always available; z2 optional (None = not available this step)."""
        self.predict()
        self._update(z1, self.H1, self.R1)
        if z2 is not None:
            self._update(z2, self.H2, self.R2)
        return self.x.copy()


def run_fusion_experiment(n_steps: int = 200, sensor2_freq: int = 5, seed: int = 42):
    """
    Compare: single sensor KF vs dual sensor fusion KF vs raw sensor 1.
    sensor2_freq: sensor 2 available every N steps (simulates lower frequency).
    """
    from kalman_filter import KalmanFilter

    np.random.seed(seed)

    true_positions, true_velocities = [], []
    obs_s1, obs_s2 = [], []

    pos, vel = 0.0, 1.0
    for _ in range(n_steps):
        vel += np.random.normal(0, 0.05)
        pos += vel
        true_positions.append(pos)
        true_velocities.append(vel)
        obs_s1.append(pos + np.random.normal(0, 2.0))   # noisy GPS
        obs_s2.append(pos + np.random.normal(0, 0.7))   # precise accel

    true_positions = np.array(true_positions)
    obs_s1 = np.array(obs_s1)
    obs_s2 = np.array(obs_s2)

    # Single sensor KF
    kf_single = KalmanFilter(dt=1.0, process_noise=0.1, measurement_noise=4.0)
    est_single = []
    for z in obs_s1:
        state = kf_single.step(z)
        est_single.append(state[0, 0])

    # Fusion KF
    kf_fusion = SensorFusionKF(dt=1.0, process_noise=0.1, noise_s1=4.0, noise_s2=0.49)
    est_fusion = []
    for i, z1 in enumerate(obs_s1):
        z2 = obs_s2[i] if i % sensor2_freq == 0 else None
        state = kf_fusion.step(z1, z2)
        est_fusion.append(state[0, 0])

    est_single = np.array(est_single)
    est_fusion = np.array(est_fusion)

    def rmse(a, b): return np.sqrt(np.mean((a - b) ** 2))

    return {
        "true_positions": true_positions,
        "obs_s1": obs_s1,
        "obs_s2": obs_s2,
        "est_single": est_single,
        "est_fusion": est_fusion,
        "rmse_raw": rmse(true_positions, obs_s1),
        "rmse_single": rmse(true_positions, est_single),
        "rmse_fusion": rmse(true_positions, est_fusion),
    }
