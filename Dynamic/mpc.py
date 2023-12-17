import numpy as np
import casadi as ca
from quadrotor import Quadrotor
import matplotlib.pyplot as plt

def mpc_control(quadrotor, N, x_init, x_target, num_states=4, num_inputs=2):
    # Create an optimization problem
    opti = ca.Opti()

    # State & Input matrix
    Q = np.eye(num_states)
    R = np.eye(num_inputs)

    # Define Variables
    x = opti.variable(num_states, N + 1)
    u = opti.variable(num_inputs, N) 

    # Initialize Cost & Constraints
    cost = 0.0
    constraints = []

    #Loop through time steps
    for k in range(N):
        # Dynamics Constraint
        constraints += [x[:, k+1] == quadrotor.calculate_next_step(x[:, k], u[:, k])]

        # Cost function
        e_k = x_target - x[:, k]
        cost += ca.mtimes(e_k.T, Q @ e_k) + ca.mtimes(u[:, k].T, R @ u[:, k])
    
    # Init Constraint
    constraints += [x[:, 0] == x_init] 

    # Cost last state
    e_N = x_target - x[:, -1]
    cost += ca.mtimes(e_N.T, Q @ e_N)

    # Define Problem in solver
    opti.minimize(cost)
    opti.subject_to(constraints)

    opts = {'ipopt.print_level': 0}  # Set the verbosity level (0-12, default is 5)
    opti.solver('ipopt', {'ipopt.print_level': 0})

    # Run Solver
    try:
        sol = opti.solve()
        optimal_solution_u = sol.value(u)
        optimal_solution_x = sol.value(x)
        optimal_cost = sol.value(cost)
        print("Optimal cost:", optimal_cost)
    except RuntimeError:
        print("Solver failed to find a solution.")

    return optimal_solution_u[:, 0], optimal_solution_x[:, 1], optimal_solution_x

x_init = np.zeros(12)
x_target = np.zeros(12)
x_target[0:3] = [0, 0, 10]

positions = mpc_control(Quadrotor(0.01), 500, x_init, x_target, 12, 4)[2][0:6, :]

# Extract x, y, z coordinates
x_positions = positions[0, :]
y_positions = positions[1, :]
z_positions = positions[2, :]
x_rot = positions[3, :]
y_rot = positions[4, :]
z_rot = positions[5, :]

# Plot the positions
plt.figure(figsize=(10, 6))
plt.plot(x_positions, label='X Position')
plt.plot(y_positions, label='Y Position')
plt.plot(z_positions, label='Z Position')
plt.plot(x_rot, label='X rot')
plt.plot(y_rot, label='Y rot')
plt.plot(z_rot, label='Z rot')
plt.xlabel('Time Step')
plt.ylabel('Position (meters)')
plt.title('Quadrotor Positions')
plt.legend()
plt.grid(True)
plt.show()