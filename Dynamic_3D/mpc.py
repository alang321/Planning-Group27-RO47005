import numpy as np
import casadi as ca

def mpc_control(vehicle, N, x_init, x_target, pos_constraints, vel_constraints, acc_constraints, obstacles = [], move_obstacles = [],  num_states=6, num_inputs=3):
    # Create an optimization problem
    print (f"\n\n CURRENT TARGET: {x_target} \n\n")

    opti = ca.Opti()
    
    # State & Input matrix
    Q = np.eye(2)
    R = np.eye(2)

    # Define Variables
    x = opti.variable(num_states, N + 1)
    u = opti.variable(num_inputs, N) 

    # Initialize Cost & Constraints
    cost = 0.0
    constraints = []

    V_obstacles = obstacles[0]
    V_move_obstacles = move_obstacles[0]
    H_obstacles = obstacles[1]
    H_move_obstacles = move_obstacles[1]

    #Loop through time steps
    for k in range(N):

        # State Constraints
        constraints += SetFixedDroneConstraints(x, u, k, pos_constraints, vel_constraints, acc_constraints)
        
        # Vertical Static Obstacle Constraints 
        SOconstraints, SOcost = VerticalStaticObstacleConstraints(V_obstacles, x, k)
        constraints += SOconstraints
        cost += SOcost

        # Vertical Moving Obstacle Constraints
        DOconstraints, DOcost = VerticalDynamicObstacleConstraints(V_move_obstacles, x, k)
        constraints += DOconstraints
        cost += DOcost

        # Horizontal Static Obstacle Constraints	
        DOconstraints, DOcost = HorizontalStaticObstacleConstraints(H_obstacles, x, k)
        constraints += DOconstraints
        cost += DOcost 

        # Horizontal Moving Obstacle Constraints
        DOconstraints, DOcost = HorizontalDynamicObstacleConstraints(H_move_obstacles, x, k)
        constraints += DOconstraints
        cost += DOcost 
        
        # Dynamics Constraint
        constraints += [x[:, k+1] == vehicle.A @ x[:, k] + vehicle.B @ u[:, k]]

        # Cost function
        e_k = x_target[:2] - x[:, k]
        cost += ca.mtimes(e_k.T, Q @ e_k) + ca.mtimes(u[:, k].T, R @ u[:, k])
    
    # Init Constraint
    constraints += [x[:, 0] == x_init] 

    # Cost last state
    e_N = x_target - x[:, -1]
    cost += ca.mtimes(e_N.T, Q @ e_N)
    
    
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
        return [0, 0, 0], x_init, None




def SetFixedDroneConstraints(x, u, k, pos_constraints, vel_constraints, acc_constraints):
    # Set fixed states and acceleration constraints
    constraints = []
    constraints += [x[:, k] >= [pos_constraints[0], pos_constraints[2], pos_constraints[4], vel_constraints[0], vel_constraints[2], vel_constraints[4]]]
    constraints += [x[:, k] <= [pos_constraints[1], pos_constraints[3], pos_constraints[5], vel_constraints[1], vel_constraints[3], vel_constraints[5]]]
    constraints += [u[:, k] >= [acc_constraints[0], acc_constraints[2], acc_constraints[4]]]
    constraints += [u[:, k] <= [acc_constraints[1], acc_constraints[3], acc_constraints[5]]]
    return constraints

def VerticalStaticObstacleConstraints(obstacles, x, k):
    constraints = []
    cost = 0
    for obstacle in obstacles:
            constraints += obstacle.get_constraint(x, k)
            cost = obstacle.get_cost(x, k, 200)
    return constraints, cost

def VerticalDynamicObstacleConstraints(move_obstacles, x, k):
    constraints = []
    cost = 0
    for obstacle in move_obstacles:
            constraints += obstacle.get_constraint(x, k)
            cost = obstacle.get_cost(x, k, 400)
    return constraints, cost

def HorizontalStaticObstacleConstraints(obstacles, x, k):
    constraints = []
    cost = 0
    for obstacle in obstacles:
            constraints += obstacle.get_constraint(x, k)
            cost = obstacle.get_cost(x, k, 200)
    return constraints, cost

def HorizontalDynamicObstacleConstraints(move_obstacles, x, k):
    constraints = []
    cost = 0
    for obstacle in move_obstacles:
            constraints += obstacle.get_constraint(x, k)
            cost = obstacle.get_cost(x, k, 400)
    return constraints, cost


