import numpy as np
import casadi as ca
from quadrotor import Quadrotor
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from helper_functions import path_ref, angle_difference, animate_trajectory, plot_control_inputs

def mpc_control(quadrotor, N, x_init, x_target, num_states=4, num_inputs=2):
    # Create an optimization problem
    opti = ca.Opti()

    # Define Variables
    x = opti.variable(num_states, N + 1)
    u = opti.variable(num_inputs, N) 

    # Initialize Cost & Constraints
    cost = 0.0
    constraints = []

    # Motor Limit
    rpm_lim = 1000
    
    # Weights
    weights = {
        "positions": 120,
        "attitudes": 20,
        "velocities": 0,
        "ang_velocities": 0,
        "inputs": 0.1
    }

    Q = ca.diagcat(
        np.eye(3) * weights["positions"],
        np.eye(3) * weights["attitudes"],
        np.eye(3) * weights["velocities"],
        np.eye(3) * weights["ang_velocities"]
    )

    R = np.eye(num_inputs) * weights["inputs"]

    
    time = 0
    #Loop through time steps
    for k in range(N):
        time += quadrotor.dt

        # Dynamics Constraint
        constraints += [x[:, k+1] == quadrotor.calculate_next_step(x[:, k], u[:, k])]

        # Motor Limits
        for i in range(4):
            opti.subject_to(opti.bounded(0, u[i, k], rpm_lim))

        # Cost function
        x_target[0:2] = path_ref(time, 3)
        e_k = x_target - x[:, k]
        e_k[3:6] = angle_difference(x_target[3:6], x[3:6, k])
        cost += ca.mtimes(e_k.T, ca.mtimes(Q, e_k)) + ca.mtimes(u[:, k].T, ca.mtimes(R, u[:, k]))

    
    # Init Constraint
    constraints += [x[:, 0] == x_init] 

    # Define Problem in solver
    opti.minimize(cost)
    opti.subject_to(constraints)

    opti.solver('ipopt', {'ipopt.print_level': 5}) # Set the verbosity level (0-12, default is 5)

    # Run Solver
    try:
        sol = opti.solve()
        optimal_solution_u = sol.value(u)
        optimal_solution_x = sol.value(x)
        optimal_cost = sol.value(cost)
        print("Optimal cost:", optimal_cost)
    except RuntimeError:
        print("Solver failed to find a solution.")

    return optimal_solution_x, optimal_solution_u

# Run simulation
dt = 0.01
x_init = np.zeros(12)
x_target = np.zeros(12)
x_target[0:3] = [0, 0, 0]
results = mpc_control(Quadrotor(dt), 200, x_init, x_target, 12, 4)

# Extract positions and control inputs
positions = results[0][0:6, :]
inputs = results[1]

plot_control_inputs(inputs)
animate_trajectory(positions, dt)
