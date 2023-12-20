import numpy as np
import casadi as ca
from vehicle import BaseVehicle

class Quadrotor(BaseVehicle):
    def __init__(self, dt):
        # System Constants
        self.dt = dt
        self.mass = 1.0  # kg
        self.gravity = -9.81  # m / s^2

        # Quadrotor Constants
        self.k_F = 1.0
        self.L = 0.2
        self.k_M = 0.1

        self.I_xx = 0.1
        self.I_yy = 0.1
        self.I_zz = 0.2

        self.I = np.diag([self.I_xx, self.I_yy, self.I_zz])

    def return_rotation_matrix(self, x):
        phi, theta, psi = ca.vertsplit(x[3:6], 1)

        # Euler ZYX angles convention
        Rx = ca.vertcat(
            ca.horzcat(1, 0, 0),
            ca.horzcat(0, ca.cos(phi), -ca.sin(phi)),
            ca.horzcat(0, ca.sin(phi), ca.cos(phi))
        )

        Ry = ca.vertcat(
            ca.horzcat(ca.cos(theta), 0, ca.sin(theta)),
            ca.horzcat(0, 1, 0),
            ca.horzcat(-ca.sin(theta), 0, ca.cos(theta))
        )

        Rz = ca.vertcat(
            ca.horzcat(ca.cos(psi), -ca.sin(psi), 0),
            ca.horzcat(ca.sin(psi), ca.cos(psi), 0),
            ca.horzcat(0, 0, 1)
        )

        return Rz @ Ry @ Rx

    def return_tau(self, u):
        L = self.L
        k_F = self.k_F
        k_M = self.k_M

        return ca.vertcat(
            L * k_F * (u[1] - u[3]),
            L * k_F * (-u[0] + u[2]),
            k_M * (-u[0] + u[1] - u[2] - u[3])
        )

    def return_F(self, u):
        k_F = self.k_F
        return ca.vertcat(0, 0, k_F * (u[0] + u[1] + u[2] + u[3]))

    def CalculateNextStep(self, x, u):
        R = self.return_rotation_matrix(x)
        F = self.return_F(u)
        tau = self.return_tau(u)
        I = ca.DM(self.I)
        v = x[6:9]
        omega = x[9:]
        v_dot = ca.vertcat(0, 0, 1) * self.gravity + ca.mtimes(R, F) * (1 / self.mass)
        omega_dot = -ca.mtimes(ca.inv(I), (ca.cross(omega, ca.mtimes(I, omega)) + tau))

        dt = self.dt
        x_next = ca.vertcat(
            x[0:3] + v * dt,
            x[3:6] + omega * dt,
            x[6:9] + v_dot * dt,
            x[9:] + omega_dot * dt
        )

        return x_next


