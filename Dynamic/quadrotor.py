import numpy as np

class quadrotor:
    def __init__(self, dt):
        # System Constants
        self.dt = dt
        self.mass = 1  # kg
        self.gravity = -9.81  # m / s^2

        # Quadrotor Constants
        self.k_F = 1
        self.L = 1
        self.k_M = 1

        self.I_xx = 1
        self.I_yy = 1
        self.I_zz = 1

        self.I = np.diag([self.I_xx, self.I_yy, self.I_zz])

    def return_rotation_matrix(self, x):
        phi, theta, psi = x[3:6]

        # Euler ZYX angles convention
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])

        Ry = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        Rz = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])

        return Rz @ Ry @ Rx

    def return_tau(self, u):
        L = self.L
        k_F = self.k_F
        k_M = self.k_M

        return np.array([
            L*k_F * (u[1] - u[3]),
            L*k_F * (-u[0] + u[2]),
            k_M * (-u[0] + u[1] - u[2] - u[3])
        ])

    def return_F(self, u):
        k_F = self.k_F
        return np.array([
            0,
            0,
            k_F * (u[0] + u[1] + u[2] + u[3])
        ])

    def calculate_next_step(self, x, u):
        R = self.return_rotation_matrix(x)
        F = self.return_F(u)
        tau = self.return_tau(u)
        I = self.I
        v = x[6:9]
        omega = x[9:]
        v_dot = np.eye(3) * -self.gravity + (R @ F) * (1 / self.mass)
        omega_dot = -np.linalg.inv(I) @ (np.cross(omega, I @ omega) + tau)

        x_dot = np.concatenate([v, omega, v_dot, omega_dot])
        return x_dot
