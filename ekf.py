"""
Extended Kalman Filter (EKF) for non-linear state estimation.
Models a pendulum system: non-linear dynamics, noisy angle observations.
"""

import numpy as np


class ExtendedKalmanFilter:
    """
    EKF for a simple pendulum (non-linear system).

    State: [theta (angle), omega (angular velocity)]
    Observation: noisy angle measurement

    Non-linear dynamics:
        theta_{k+1} = theta_k + omega_k * dt
        omega_{k+1} = omega_k - (g/L) * sin(theta_k) * dt
    """

    def __init__(self, dt: float = 0.05, g: float = 9.81, L: float = 1.0,
                 process_noise: float = 0.01, measurement_noise: float = 0.1):
        self.dt = dt
        self.g = g
        self.L = L

        self.Q = process_noise * np.eye(2)
        self.R = np.array([[measurement_noise]])

        self.x = np.array([[0.5], [0.0]])   # initial angle=0.5 rad, omega=0
        self.P = np.eye(2) * 0.1

    def f(self, x):
        """Non-linear state transition."""
        theta, omega = x[0, 0], x[1, 0]
        theta_new = theta + omega * self.dt
        omega_new = omega - (self.g / self.L) * np.sin(theta) * self.dt
        return np.array([[theta_new], [omega_new]])

    def F_jacobian(self, x):
        """Jacobian of f w.r.t. state (linearization)."""
        theta = x[0, 0]
        return np.array([
            [1, self.dt],
            [-(self.g / self.L) * np.cos(theta) * self.dt, 1]
        ])

    def h(self, x):
        """Observation model: observe angle only."""
        return np.array([[x[0, 0]]])

    def H_jacobian(self):
        return np.array([[1, 0]])

    def predict(self):
        F = self.F_jacobian(self.x)
        self.x = self.f(self.x)
        self.P = F @ self.P @ F.T + self.Q
        return self.x.copy()

    def update(self, z: float):
        z = np.array([[z]])
        H = self.H_jacobian()
        y = z - self.h(self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ H) @ self.P
        return self.x.copy()

    def step(self, z: float):
        self.predict()
        return self.update(z)
