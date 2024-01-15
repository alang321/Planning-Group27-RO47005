import numpy as np
import casadi as ca
import helper_functions


def mpc_control(vehicle, N, x_init, x_target, pos_constraints, vel_constraints, acc_constraints, ang_constraints,
                max_rad_per_s, last_plan, obstacles=[], num_states=12, num_inputs=4):
    # Create an optimization problem

    opti = ca.Opti()

    # State & Input weight matrix
    Q = np.eye(3) * 4
    R = np.eye(num_inputs) * 0.001

    Q_vel = np.eye(3) * 0.7
    Q_angle = np.eye(3) * 0.05

    # Define Variables
    x = opti.variable(num_states, N + 1)
    u = opti.variable(num_inputs, N)

    # Initialize Cost
    cost = 0.0

    # Loop through time steps
    for k in range(N):
        # Calculate Acceleration
        v_dot = vehicle.get_v_dot(x[:, k], u[:, k])


        # State Constraints
        opti.subject_to(
            SetFixedDroneConstraints(x, u, v_dot, k, pos_constraints, acc_constraints, ang_constraints, vel_constraints,
                                     max_rad_per_s))

        # Vertical Static Obstacle Constraints
        SOconstraints, SOcost = StaticObstacleConstraints(obstacles, x, k + 1)
        if SOconstraints:
            opti.subject_to(SOconstraints)
        cost += SOcost

        # Dynamics Constraint
        opti.subject_to([x[:, k + 1] == vehicle.calculate_next_step(x[:, k], u[:, k])])

        #cost for attitude
        error = helper_functions.return_angle_difference(x_target, x[:, k])
        cost += ca.mtimes(error.T, Q_angle @ error)

        #cost for velocity
        error = helper_functions.return_vel_difference(x_target, x[:, k])
        cost += ca.mtimes(error.T, Q_vel @ error)

        # Cost function
        error = helper_functions.return_pos_diff(x_target, x[:, k])
        cost += ca.mtimes(error.T, Q @ error) + ca.mtimes(u[:, k].T, R @ u[:, k])

    # Initial Constraint
    opti.subject_to([x[:, 0] == x_init])

    # Cost last state
    error = helper_functions.return_vel_difference(x_target, x[:, N])
    cost += ca.mtimes(error.T, Q_vel @ error)
    error = helper_functions.return_angle_difference(x_target, x[:, N])
    cost += ca.mtimes(error.T, Q_angle @ error)
    error = helper_functions.return_pos_diff(x_target, x[:, N])
    cost += ca.mtimes(error.T, Q @ error)

    # Warm start
    if last_plan is not None:
        opti.set_initial(x, last_plan)
    else:
        # Setup initial guess for states (interpolate between x_init and x_target)
        for k in range(N + 1):
            x_guess = x_init + (x_target - x_init) * k / N
            #x_guess[6] = 0  # Set initial guess for speed in z direction to zero
            opti.set_initial(x[:, k], x_guess)

    # Define Problem in solver
    opti.minimize(cost)
    opti.solver('ipopt', {'ipopt.print_level': 0, 'ipopt.tol': 1e-6, 'ipopt.acceptable_tol': 1e-4})

    # Run Solver
    try:
        sol = opti.solve()
        optimal_solution_u = sol.value(u)
        optimal_solution_x = sol.value(x)
        optimal_cost = sol.value(cost)
        print("Optimal cost:", optimal_cost)
        return optimal_solution_u[:, 0], optimal_solution_x[:, 1], optimal_solution_x
    except:
        print("Solver failed. Infeasible or no solution found.")
        print("Failed constraints:")
        opti.debug.show_infeasibilities()
        return [0, 0, 0, 0], x_init, None


def SetFixedDroneConstraints(x, u, v_dot, k, pos_constraints, vel_constraints, acc_constraints, ang_constraints,
                             max_rad_per_s):
    # Set fixed states and acceleration constraints
    ang_constraints = np.deg2rad(ang_constraints)
    constraints = []

    # State constraints
    constraints += [
        x[:, k] >= [pos_constraints[0], pos_constraints[2], pos_constraints[4], ang_constraints[0], ang_constraints[2],
                    ang_constraints[4], vel_constraints[0], vel_constraints[2], vel_constraints[4], -10000, -10000,
                    -10000]]
    constraints += [
        x[:, k] <= [pos_constraints[1], pos_constraints[3], pos_constraints[5], ang_constraints[1], ang_constraints[3],
                    ang_constraints[5], vel_constraints[1], vel_constraints[3], vel_constraints[5], 10000, 10000,
                    10000]]

    # Acceleration constraints
    constraints += [v_dot[0] >= acc_constraints[0]]
    constraints += [v_dot[0] <= acc_constraints[1]]
    constraints += [v_dot[1] >= acc_constraints[2]]
    constraints += [v_dot[1] <= acc_constraints[3]]
    constraints += [v_dot[2] >= acc_constraints[4]]
    constraints += [v_dot[2] <= acc_constraints[5]]

    # # Control input constraints
    constraints += [u[:, k] >= [0, 0, 0, 0]]
    constraints += [u[:, k] <= [max_rad_per_s ** 2, max_rad_per_s ** 2, max_rad_per_s ** 2, max_rad_per_s ** 2]]
    return constraints


def StaticObstacleConstraints(obstacles, x, k):
    constraints = []
    cost = 0
    for obstacle in obstacles:
        constraints += obstacle.get_constraint(x, k)

        cost += obstacle.get_cost(x, k)
    return constraints, cost



