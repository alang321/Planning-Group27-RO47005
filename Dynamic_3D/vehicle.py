import numpy as np

class vehicle_SS:
    def __init__(self, dt):
        # State space matrices
        self.A_c = np.matrix([[0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0]])

        self.B_c = np.matrix([[0,  0, 0], 
                              [0,  0, 0],
                              [0,  0, 0],
                              [dt, 0, 0],
                              [0, dt, 0],
                              [0,  0, dt]])

        # self.C_c = np.matrix([[1, 0, 0, 0],
        #             [0, 1, 0, 0]])

        # self.D_c = np.matrix([[0, 0],
        #             [0, 0]])

        self.A = np.eye(6) + self.A_c * dt
        self.B = self.B_c
        # self.C = self.C_c
        # self.D = self.D_c

    def CalculateNextStep(self, x, u):
        x_next = self.A.dot(x.T) + self.B.dot(u.T)
        return x_next
    

import numpy as np
import casadi as ca

class Quadrotor:
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
    # def euler_to_quaternion(self, phi, theta, psi):
    #     """
    #     Convert Euler angles to quaternion.
    #     """
    #     cy = ca.cos(psi * 0.5)
    #     sy = ca.sin(psi * 0.5)
    #     cp = ca.cos(theta * 0.5)
    #     sp = ca.sin(theta * 0.5)
    #     cr = ca.cos(phi * 0.5)
    #     sr = ca.sin(phi * 0.5)
    #
    #     q_w = cr * cp * cy + sr * sp * sy
    #     q_x = sr * cp * cy - cr * sp * sy
    #     q_y = cr * sp * cy + sr * cp * sy
    #     q_z = cr * cp * sy - sr * sp * cy
    #
    #     return [q_w, q_x, q_y, q_z]
    #
    # def return_rotation_matrix(self, q):
    #     """
    #     Convert quaternion to rotation matrix.
    #     """
    #     q_w, q_x, q_y, q_z = q
    #
    #     R = ca.horzcat(
    #         ca.horzcat(1 - 2 * (q_y ** 2 + q_z ** 2), 2 * (q_x * q_y - q_w * q_z), 2 * (q_x * q_z + q_w * q_y)),
    #         ca.horzcat(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_x ** 2 + q_z ** 2), 2 * (q_y * q_z - q_w * q_x)),
    #         ca.horzcat(2 * (q_x * q_z - q_w * q_y), 2 * (q_y * q_z + q_w * q_x), 1 - 2 * (q_x ** 2 + q_y ** 2))
    #     )
    #
    #     return R

    def return_tau(self, u):
        L = self.L
        k_F = self.k_F
        k_M = self.k_M

        return ca.vertcat(
            L * k_F * (u[1] - u[3]),
            L * k_F * (-u[0] + u[2]),
            k_M * (-u[0] + u[1] - u[2] + u[3])
        )

    def return_F(self, u):
        k_F = self.k_F
        return ca.vertcat(0, 0, k_F * (u[0] + u[1] + u[2] + u[3]))

    def calculate_next_step(self, x, u):
        R = self.return_rotation_matrix(x)

        #TODO: fix or revert quaternion method
        #quaternion method
        # q = self.euler_to_quaternion(x[3], x[4], x[5])  # Convert Euler angles to quaternion
        # R = self.return_rotation_matrix(q)  # Convert quaternion to rotation matrix


        F = self.return_F(u)
        tau = self.return_tau(u)
        I = ca.DM(self.I)
        v = x[6:9]
        omega = x[9:]
        v_dot = ca.vertcat(0, 0, 1) * self.gravity + ca.mtimes(R, F) * (1 / self.mass)
        omega_dot = ca.mtimes(ca.inv(I), (ca.cross(-omega, ca.mtimes(I, omega)) + tau))

        dt = self.dt
        x_next = ca.vertcat(
            x[0:3] + v * dt,
            x[3:6] + omega * dt,
            x[6:9] + v_dot * dt,
            x[9:] + omega_dot * dt
        )

        return x_next


