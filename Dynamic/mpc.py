import numpy as np
import casadi as ca

def mpc_control(vehicle, N, x_init, x_target, pos_constraints, vel_constraints, acc_constraints, obstacles = [], move_obstacles = [],  num_states=4, num_inputs=2):
    # Create an optimization problem
    print (f"\n\n CURRENT TARGET: {x_target} \n\n")

    opti = ca.Opti()


    # State & Input matrix
    Q = np.array([[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    R = np.eye(2)

    # Define Variables
    x = opti.variable(num_states, N + 1)
    u = opti.variable(num_inputs, N) 

    # Initialize Cost & Constraints
    cost = 0.0
    constraints = []

    #Loop through time steps
    for k in range(N):
        # State Constraints
        constraints += [x[:, k] >= [pos_constraints[0], pos_constraints[2], vel_constraints[0], vel_constraints[2]]]
        constraints += [x[:, k] <= [pos_constraints[1], pos_constraints[3], vel_constraints[1], vel_constraints[3]]]
        constraints += [u[:, k] >= [acc_constraints[0], acc_constraints[2]]]
        constraints += [u[:, k] <= [acc_constraints[1], acc_constraints[3]]]
        
        for obstacle in obstacles:
            euclid_distance = ca.norm_2( x[0:2,k] - np.array(obstacle[0:2]).reshape(2,1) )
            constraints += [euclid_distance >= obstacle[2]]
            cost += 200/((euclid_distance-obstacle[2])**2 + 0.01)
            
        for obstacle in move_obstacles:
            euclid_distance = ca.norm_2( x[0:2,k] - np.array(obstacle[0:2]).reshape(2,1) )
            constraints += [euclid_distance > obstacle[2]]
            cost += 400/((euclid_distance-obstacle[2])**2 + 0.01)

        # Dynamics Constraint
        constraints += [x[:, k+1] == vehicle.A @ x[:, k] + vehicle.B @ u[:, k]]

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
        return optimal_solution_u[:, 0], optimal_solution_x[:, 1], optimal_solution_x
    except RuntimeError:
        print("Solver failed to find a solution.")
        return [0, 0], x_init, None