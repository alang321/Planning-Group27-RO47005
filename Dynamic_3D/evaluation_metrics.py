import numpy as np
import matplotlib.pyplot as plt

### Functions for evaluation metrics and plotting

# 1. Trajectory Smoothness
def get_trajectory_smoothness(velocities, dt):
    # calculate acceleration
    accelerations = np.diff(velocities, axis=0) / dt
    # calculate jerk
    jerks = np.diff(accelerations, axis=0) / dt

    # Print shapes for verification
    print("accelerations:", accelerations.shape)
    print("jerks:", jerks.shape)

    return accelerations, jerks

def plot_trajectory_smoothness(velocities, orientation_rates, num_timesteps, duration_sec, dt):
    position_accelerations, position_jerks = get_trajectory_smoothness(velocities, dt)
    orientation_accelerations, orientation_jerks = get_trajectory_smoothness(orientation_rates, dt)
    time = np.linspace(0, duration_sec, num_timesteps)
    print("time shape:", time.shape)
    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot Position Acceleration
    plt.subplot(2, 2, 1)
    plt.plot(time, position_accelerations[:, 0], label='X Acceleration')
    plt.plot(time, position_accelerations[:, 1], label='Y Acceleration')
    plt.plot(time, position_accelerations[:, 2], label='Z Acceleration')
    plt.title('Position Acceleration Over Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s^2]')
    plt.legend()

    # Plot Position Jerk
    plt.subplot(2, 2, 2)
    plt.plot(time[:-1], position_jerks[:, 0], label='X Jerk')
    plt.plot(time[:-1], position_jerks[:, 1], label='Y Jerk')
    plt.plot(time[:-1], position_jerks[:, 2], label='Z Jerk')
    plt.title('Position Jerk Over Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Jerk [m/s^3]')
    plt.legend()

    # Plot Orientation Acceleration
    plt.subplot(2, 2, 3)
    plt.plot(time, orientation_accelerations[:, 0], label='Roll Acceleration')
    plt.plot(time, orientation_accelerations[:, 1], label='Pitch Acceleration')
    plt.plot(time, orientation_accelerations[:, 2], label='Yaw Acceleration')
    plt.title('Orientation Acceleration Over Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [rad/s^2]') #TODO: is this the correct unit or is it rpm?
    plt.legend()

    # Plot Orientation Jerk
    plt.subplot(2, 2, 4)
    plt.plot(time[:-1], orientation_jerks[:, 0], label='Roll Jerk')
    plt.plot(time[:-1], orientation_jerks[:, 1], label='Pitch Jerk')
    plt.plot(time[:-1], orientation_jerks[:, 2], label='Yaw Jerk')
    plt.title('Orientation Jerk Over Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Jerk [rad/s^3]')
    plt.legend()

    plt.tight_layout()
    plt.show()

# 2. Control Effort
def get_control_effort(control_inputs):
    total_control_per_step = np.zeros(len(control_inputs))
    for i in range(len(control_inputs)):
        total_control_per_step[i] = np.sum(control_inputs[i])
    return total_control_per_step

def plot_control_effort(control_inputs, duration_sec, num_timesteps):
    total_control_per_step = get_control_effort(control_inputs)
    control_input_1 = control_inputs[:,0]
    control_input_2 = control_inputs[:,1]
    control_input_3 = control_inputs[:,2]
    control_input_4 = control_inputs[:,3]
    time = np.linspace(0, duration_sec, num_timesteps)
    plt.figure(figsize=(10, 8))

    # Plot Control Inputs
    plt.subplot(2, 1, 1)
    plt.plot(time, control_input_1, label='Rotor 1')
    plt.plot(time, control_input_2, label='Rotor 2')
    plt.plot(time, control_input_3, label='Rotor 3')
    plt.plot(time, control_input_4, label='Rotor 4')
    plt.title('Control Inputs Over Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Control Input [rad/s]^2')
    plt.legend()

    # Plot Total Control Input
    plt.subplot(2, 1, 2)
    plt.plot(time, total_control_per_step, label='Total Control Input', color='red')
    plt.title('Total Control Input Over Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Total Control Input [rad/s]^2')
    plt.legend()

    plt.tight_layout()
    plt.show()

# 3. Time to goal
def plot_time_to_goal(waypoints, drone_positions, goal_time, cylinder_paths):
    # overhead plot
    # path drone took (green)
    # plot cylinder paths (red)
    # plot & number waypoints (blue)
    # chart of duration between each waypoint & total time
    # Plotting
    plt.figure(figsize=(10, 8))

    # Plot Drone Path (green)
    plt.plot(drone_positions[:, 0], drone_positions[:, 1], label='Drone Path', color='green')

    # Plot & Number Waypoints (blue)
    for i, waypoint in enumerate(waypoints):
        plt.scatter(waypoint[0], waypoint[1], marker='o', color='blue', label='Waypoints')
        plt.text(waypoint[0], waypoint[1], f'{i + 1}', fontsize=12, ha='right', va='top')
        # plt.text(waypoint[0], waypoint[1], f'{waypoint_time_markers[i]}', fontsize=12, ha='right', va='bottom')
        if i == len(waypoints):
            plt.text(waypoint[0], waypoint[1], f'{goal_time}', fontsize=12, ha='right', va='bottom')

    # Plot Cylinder Paths
    if cylinder_paths:
        for path in cylinder_paths:
            plt.plot(path[:, 0], path[:, 1], linestyle='--', color='red', label='Cylinder Path')

    plt.title('Drone Path with Waypoints and Cylinder Paths')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

# 4. Computation time
# get total time for rrt*
# get total time for mpc