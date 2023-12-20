import numpy as np
def MovingObstacleConvert(move_obstacles, t, dt):
    new_fixed_obs = []
    for obstacle in move_obstacles:
        new_fixed = []
        new_fixed.append(obstacle[0] + (t*dt*obstacle[2])) # X Position
        new_fixed.append(obstacle[1] + (t*dt*obstacle[3])) # Y Position
        new_fixed.append(obstacle[-1])                           # Radius 
        new_fixed_obs.append(new_fixed)

    return new_fixed_obs

def simulate(dt, T, x_init, plan_length, control_func, move_obstacles, path_rrt, waypoint_radius, num_states = 4, num_inputs = 2):
    ## Timesteps
    timesteps = np.arange(0, T, dt)
    print(f"Timesteps: {len(timesteps)}")

    # Initialise the output arrays
    x_real = np.zeros((num_states, len(timesteps)+1))
    x_all = np.zeros((num_states, plan_length+1, len(timesteps)+1))
    u_real = np.zeros((num_inputs, len(timesteps)))
    targets = np.zeros((num_states, len(timesteps)+1))
    
    waypoint_idx = 0
    x_real[:, 0] = x_init
    for t in range(len(timesteps)):

        # Update Waypoint
        current_state = x_real[:, t]
        distance_2_next_wp = np.sqrt((current_state[0]-path_rrt.path_points[waypoint_idx][0])**2 + (current_state[1]-path_rrt.path_points[waypoint_idx][1])**2)
        if distance_2_next_wp < waypoint_radius and waypoint_idx < len(path_rrt.path_points)-1:

            waypoint_idx += 1
     

        target_state = [path_rrt.path_points[waypoint_idx][0], path_rrt.path_points[waypoint_idx][1], 0, 0]
        # Compute the control input (and apply it)
        move_obstacles_update = MovingObstacleConvert(move_obstacles, t, dt)
        u_out, x_out, x_all_out = control_func(x_real[:, t], target_state, move_obstacles_update)

        # Next x is the x in the second state
        x_real[:, t+1] = x_out
        x_all[:, :, t] = x_all_out # Save the plan (for visualization)

        # Used input is the first input
        u_real[:, t] = u_out
        targets[:, t] = target_state 

    return x_real, u_real, x_all, timesteps, targets