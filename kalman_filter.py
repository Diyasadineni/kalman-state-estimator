"""
Kalman Filter - State Estimator
Tracks dynamic system states from noisy sensor readings.
"""

import numpy as np


class KalmanFilter:
    """
    Linear Kalman Filter for state estimation.

    State vector: [position, velocity]
    Observation: [position]
    """

    def __init__(self, dt: float = 1.0, process_noise: float = 0.1, measurement_noise: float = 1.0):
        self.dt = dt

        # State transition matrix (constant velocity model)
        self.F = np.array([[1, dt],
                           [0, 1]])

        # Observation matrix (we only observe position)
        self.H = np.array([[1, 0]])

        # Process noise covariance
        self.Q = process_noise * np.array([[dt**4 / 4, dt**3 / 2],
                                            [dt**3 / 2, dt**2]])

        # Measurement noise covariance
        self.R = np.array([[measurement_noise]])

        # Initial state estimate and covariance
        self.x = np.zeros((2, 1))       # [position, velocity]
        self.P = np.eye(2) * 1000       # High uncertainty initially

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: float):
        z = np.array([[z]])
        y = z - self.H @ self.x                          # Innovation
        S = self.H @ self.P @ self.H.T + self.R          # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)         # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x.copy()

    def step(self, z: float):
        self.predict()
        return self.update(z)
