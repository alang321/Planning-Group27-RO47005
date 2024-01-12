import numpy as np
import time

def simulate(dt, T, x_init, x_target, plan_length, control_func, world, path_rrt, waypoint_radius, num_states = 6, num_inputs = 3):
    ## Timesteps
    timesteps = np.arange(0, T, dt)

    # print(F" X_target: {x_target}")
    # Initialise the output arrays
    x_real = np.zeros((num_states, len(timesteps)+1))
    x_all = np.zeros((num_states, plan_length+1, len(timesteps)+1))
    u_real = np.zeros((num_inputs, len(timesteps)))
    targets = np.zeros((num_states, len(timesteps)+1))
    
    waypoint_idx = 0
    x_real[:, 0] = x_init
    last_plan = None
    last_input = None
    target_state = [path_rrt.path_points[waypoint_idx][0], path_rrt.path_points[waypoint_idx][1], path_rrt.path_points[waypoint_idx][2], 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # target_state = x_target
    start_time = time.time()
    for t in range(len(timesteps)):

        # Update Waypoint
        current_state = x_real[:, t]
        distance_2_next_wp = np.sqrt((current_state[0]-path_rrt.path_points[waypoint_idx][0])**2 + (current_state[1]-path_rrt.path_points[waypoint_idx][1])**2)
        if distance_2_next_wp < waypoint_radius and waypoint_idx < len(path_rrt.path_points)-1:
            waypoint_idx += 1
            target_state = [path_rrt.path_points[waypoint_idx][0], path_rrt.path_points[waypoint_idx][1], path_rrt.path_points[waypoint_idx][2], 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        if waypoint_idx == len(path_rrt.path_points):
            target_state = x_target
        # Compute the control input (and apply it)
        world.update(dt)
        u_out, x_out, x_all_out = control_func(x_real[:, t], target_state, last_input, last_plan)
        print(f"Progress: {t}/{len(timesteps)}")
        last_plan = x_all_out
        last_input = u_out

        # Next x is the x in the second state
        x_real[:, t+1] = x_out
        x_all[:, :, t] = x_all_out # Save the plan (for visualization)

        # Used input is the first input
        u_real[:, t] = u_out
        targets[:, t] = target_state 
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Average Computation time per step: {elapsed_time/len(timesteps)} seconds ")

    return x_real, u_real, x_all, timesteps, targets