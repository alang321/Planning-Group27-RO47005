import numpy as np
import casadi as ca
from quadrotor import Quadrotor
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from helper_functions import path_ref, angle_difference, animate_trajectory, plot_control_inputs

matplotlib.rcParams['animation.ffmpeg_path'] = "C:\\ffmpeg\\bin\\ffmpeg.exe"  # Replace this with the actual path

def quadrotor_mpc_control(quadrotor, N, x_init, x_target, last_plan, obstacles = [], move_obstacles = [], num_states=4, num_inputs=2):
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
    
    # Weight constants
    weights = {
        "positions": 120,
        "attitudes": 20,
        "velocities": 0,
        "ang_velocities": 0,
        "inputs": 0.1
    }

    # State weight matrix
    Q = ca.diagcat(
        np.eye(3) * weights["positions"],
        np.eye(3) * weights["attitudes"],
        np.eye(3) * weights["velocities"],
        np.eye(3) * weights["ang_velocities"]
    )

    # Input weight matrix
    R = np.eye(num_inputs) * weights["inputs"]

    #Loop through time steps
    for k in range(N):
        # Vertical Static Obstacle Constraints 
        SOconstraints, SOcost = StaticObstacleConstraints(obstacles, x, k+1)
        constraints += SOconstraints
        cost += SOcost

        # Horizontal Moving Obstacle Constraints
        DOconstraints, DOcost = DynamicObstacleConstraints(move_obstacles, x, k+1)
        constraints += DOconstraints
        cost += DOcost

        # Dynamics Constraint
        constraints += [x[:, k+1] == quadrotor.CalculateNextStep(x[:, k], u[:, k])]

        # Motor Limits
        for i in range(4):
            opti.subject_to(opti.bounded(0, u[i, k], rpm_lim))

        # Cost function
        e_k = x_target - x[:, k]  # Position error
        e_k[3:6] = angle_difference(x_target[3:6], x[3:6, k])  # Attitude error
        cost += ca.mtimes(e_k.T, ca.mtimes(Q, e_k)) + ca.mtimes(u[:, k].T, ca.mtimes(R, u[:, k])) 

    # Init Constraint
    constraints += [x[:, 0] == x_init] 

    # Define Problem in solver
    opti.minimize(cost)
    opti.subject_to(constraints)

    if last_plan is not None:
        opti.set_initial(x, last_plan)

    opti.solver('ipopt', {'ipopt.print_level': 5}) # Set the verbosity level (0-12, default is 5)

    # Run Solver
    try:
        sol = opti.solve()
        optimal_solution_u = sol.value(u)
        optimal_solution_x = sol.value(x)
        optimal_cost = sol.value(cost)
        print("Optimal cost:", optimal_cost)
        return optimal_solution_u[:, 0], optimal_solution_x[:, 1], optimal_solution_x
    except RuntimeError:
        print("Solver failed to find a solution.")
        return [0, 0, 0], x_init, None

def StaticObstacleConstraints(obstacles, x, k):
    constraints = []
    cost = 0
    for obstacle in obstacles:
            constraints += obstacle.get_constraint(x, k)
            cost += obstacle.get_cost(x, k)
    return constraints, cost

def DynamicObstacleConstraints(move_obstacles, x, k):
    constraints = []
    cost = 0
    for obstacle in move_obstacles:
            constraints += obstacle.get_constraint(x, k)
            cost += obstacle.get_cost(x, k)
    return constraints, cost

# # Simulation Setup
# dt = 0.01
# x_init = np.zeros(12)
# x_target = np.zeros(12)
# x_target[0:3] = [0, 0, 5]

# # Run simulation
# results = quadrotor_mpc_control(Quadrotor(dt), 500, x_init, x_target, None, obstacles = [], move_obstacles = [], num_states=12, num_inputs=4)