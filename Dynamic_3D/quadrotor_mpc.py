import numpy as np
import casadi as ca
from quadrotor import Quadrotor
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from helper_functions import path_ref, angle_difference, animate_trajectory, plot_control_inputs

matplotlib.rcParams['animation.ffmpeg_path'] = "C:\\ffmpeg\\bin\\ffmpeg.exe"  # Replace this with the actual path

def quadrotor_mpc_control(quadrotor,
                           N, x_init, 
                           x_target, 
                           last_plan, 
                           pos_constraints = [], 
                           rpm_limit = 1000,  
                           cost_weights = [120, 20, 0, 0, 0.1], # Cost Weights [pos, att, vel, ang_vel, inputs]
                           obstacles = [], 
                           move_obstacles = [], 
                           num_states=12, 
                           num_inputs=4,
                           depth = 0):
    
    # Create an optimization problem
    opti = ca.Opti()

    # Define Variables
    x = opti.variable(num_states, N + 1)
    u = opti.variable(num_inputs, N) 

    # Initialize Cost & Constraints
    cost = 0.0
    constraints = []

    # Motor Limit
    rad_limit = rpm_limit * 2 * np.pi / 60
    input_limit = rad_limit ** 2

    # State weight matrix
    Q = ca.diagcat(
        np.eye(3) * cost_weights[0],
        np.eye(3) * cost_weights[1],
        np.eye(3) * cost_weights[2],
        np.eye(3) * cost_weights[3]
    )

    # Input weight matrix
    R = np.eye(num_inputs) * cost_weights[4]

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
        constraints += [x[:, k+1] == quadrotor.calculate_next_step(x[:, k], u[:, k])]

        # Motor Limits
        for i in range(4):
            opti.subject_to(opti.bounded(0, u[i, k], input_limit))

        # Position Limits
        if pos_constraints != []:
            opti.subject_to(opti.bounded(pos_constraints[0], x[0, k], pos_constraints[1]))
            opti.subject_to(opti.bounded(pos_constraints[2], x[1, k], pos_constraints[3]))
            opti.subject_to(opti.bounded(pos_constraints[4], x[2, k], pos_constraints[5]))

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

    opti.solver('ipopt', {'ipopt.print_level': 0}) # Set the verbosity level (0-12, default is 5)

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
        
        # If the solver fails, return a hover command
        x_target = np.concatenate((x_init[:3], [0, 0, x_init[5]], np.zeros(6)))
        depth += 1

        if depth > 5:
            print("I GIVE UP")
            return [0, 0, 0, 0], x_init, None
        return quadrotor_mpc_control(quadrotor,
                           N, x_init, 
                           x_target, 
                           last_plan, 
                           pos_constraints, 
                           rpm_limit,  
                           cost_weights,
                           obstacles, 
                           move_obstacles, 
                           num_states, 
                           num_inputs,
                           depth)

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