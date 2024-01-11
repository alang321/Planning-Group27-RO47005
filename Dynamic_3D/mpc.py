import numpy as np
import casadi as ca

def mpc_control(vehicle, N, x_init, x_target, pos_constraints, vel_constraints, ang_constraints, max_rad_per_s, last_input, last_plan, obstacles = [], move_obstacles = [],  num_states=12, num_inputs=4):

    # Create an optimization problem
    opti = ca.Opti()
    
    # State & Input weight matrix
    Q = np.zeros((12, 12))
    Q[0:3, 0:3] = 3*np.eye(3)
    R = np.eye(num_inputs) * 1e-15

    # Define Variables
    x = opti.variable(num_states, N + 1)
    u = opti.variable(num_inputs, N) 

    # Initialize Cost & Constraints
    cost = 0.0
    constraints = []

    #Loop through time steps
    for k in range(N):

        # State Constraints
        constraints += SetFixedDroneConstraints(x, u, k, pos_constraints, ang_constraints, vel_constraints, max_rad_per_s)
        
        # Vertical Static Obstacle Constraints 
        SOconstraints, SOcost = StaticObstacleConstraints(obstacles, x, k+1)
        constraints += SOconstraints
        cost += SOcost

        # Horizontal Moving Obstacle Constraints
        DOconstraints, DOcost = DynamicObstacleConstraints(move_obstacles, x, k+1)
        constraints += DOconstraints
        cost += DOcost

        # Dynamics Constraint
        constraints += [x[:, k+1] == vehicle.calculate_next_step(x[:, k], u[:, k])]

        # Cost function
        e_k = x_target - x[:, k]
        cost += ca.mtimes(e_k.T, Q @ e_k) + ca.mtimes(u[:, k].T, R @ u[:, k])
    
    # Initial Constraint
    constraints += [x[:, 0] == x_init]
    # Terminal Constraint
    constraints += [x[3:, N] == np.zeros((9, 1))] 

    # Cost last state
    e_N = x_target - x[:, N]
    cost += ca.mtimes(e_N.T, Q @ e_N)
    
    # Warm start
    if last_plan is not None:
        opti.set_initial(x, last_plan)

    # Define Problem in solver
    opti.minimize(cost)
    opti.subject_to(constraints)
    opti.solver('ipopt', {'ipopt.print_level': 0})

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
        return [0, 0, 0, 0], x_init, None


def SetFixedDroneConstraints(x, u, k, pos_constraints, vel_constraints, ang_constraints, max_rad_per_s):
    # Set fixed states and acceleration constraints
    ang_constraints = np.deg2rad(ang_constraints)
    constraints = []

    # State constraints
    constraints += [x[:, k] >= [pos_constraints[0], pos_constraints[2], pos_constraints[4], ang_constraints[0], ang_constraints[2], ang_constraints[4], vel_constraints[0], vel_constraints[2], vel_constraints[4], -100, -100, -100]]
    constraints += [x[:, k] <= [pos_constraints[1], pos_constraints[3], pos_constraints[5], ang_constraints[1], ang_constraints[3], ang_constraints[5], vel_constraints[1], vel_constraints[3], vel_constraints[5], 100, 100, 100]]
    
    # Control input constraints
    constraints += [u[:, k] >= [0, 0, 0, 0]]
    constraints += [u[:, k] <= [max_rad_per_s**2, max_rad_per_s**2, max_rad_per_s**2, max_rad_per_s**2]]
    return constraints

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



