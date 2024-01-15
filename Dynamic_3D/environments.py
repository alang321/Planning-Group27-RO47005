import numpy as np
from obstacles import *

class EnvironmentMaze:
    def __init__(self, static_cost, dynamic_cost):
        self.start = [10, 10, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Initial State: [x, y, z, x_dot, y_dot, z_dot]
        self.goal = [5, 55, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.V_obstacles = [[2, 20, 2], [6, 20, 2], [10, 20, 2], [14, 20, 2], [18, 20, 2], [22, 20, 2], [22, 16, 2],
                       [22, 12, 2], [22, 8, 2],  ## First Box
                       [58, 15, 2], [54, 15, 2], [50, 15, 2], [46, 15, 2], [42, 15, 2], [38, 15, 2],  ## Top wall
                       [10, 58, 2], [10, 54, 2], [10, 50, 2], [10, 46, 2], [10, 42, 2], [10, 38, 2], [10, 34, 2],
                       [10, 30, 2],  ## Last Wall
                       [38, 19, 2], [38, 23, 2], [38, 27, 2], [42, 27, 2], [46, 27, 2], [50, 27, 2],
                       [22, 24, 2], [22, 28, 2], [22, 32, 2], [22, 36, 2], [22, 40, 2], [22, 44, 2], [26, 40, 2],
                       [30, 40, 2], [34, 40, 2], [38, 40, 2], [42, 40, 2], [46, 40, 2], [50, 40, 2], [54, 40, 2],
                       [50, 43, 1], [54, 47, 1], [48, 54, 1], [35, 50, 3], [37, 58, 1], [28, 47, 1], [39, 49, 1],
                       [42, 58, 1], [45, 46, 2], [23, 55, 4], [30, 30, 3]]
        self.V_move_obstacles = [[30, 0, 0, 0.5, 2, 8], [60, 35, -1, 0, 2, 5], [60, 60, -0.6, -0.2, 2, 25],
                            [40, 60, 0, -0.5, 2, 20]]  # [center_x, center_y, vel_x, vel_y, radius]
        # V_move_obstacles = []
        self.H_obstacles = []  # [center_y, center_z, radius]
        self.H_move_obstacles = []  # [center_y, center_z, vel_y, vel_z, radius]

        self.static_cost = static_cost
        self.dynamic_cost = dynamic_cost

        self.obstacles = init_obstacles(self.V_obstacles, self.V_move_obstacles, self.H_obstacles, self.H_move_obstacles, self.static_cost,
                                   self.dynamic_cost)

        self.pos_constraints = [0, 60, 0, 60, 0, 20]
        self.simulation_time = 80

    def reset(self):
        self.start = init_obstacles(self.V_obstacles, self.V_move_obstacles, self.H_obstacles, self.H_move_obstacles, self.static_cost,
                                   self.dynamic_cost)


class EnvironmentOriginal:
    def __init__(self, static_cost, dynamic_cost):
        self.start = [10, 2, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Initial State: [x, y, z, x_dot, y_dot, z_dot]
        self.goal = [10, 48, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        ## Define Obstacles
        # V_obstacles = [[10, 20, 4], [18, 30, 3]]
        self.V_obstacles = [[10, 30, 2], [17, 23, 3], [7, 20, 4], [14, 37, 3], [1, 28, 4]]
        self.V_move_obstacles = [[5, 30, 1, 0, 1.5, 8], [12, 45, 0, -1, 2, 30],
                            [3, 15, 1, 0, 3, 8]]  # [center_x, center_y, vel_x, vel_y, radius]
        # V_move_obstacles = []
        self.H_obstacles = [[10, 10, 3]]  # [center_y, center_z, radius]
        self.H_move_obstacles = []  # [center_y, center_z, vel_y, vel_z, radius]

        self.static_cost = static_cost
        self.dynamic_cost = dynamic_cost

        self.obstacles = init_obstacles(self.V_obstacles, self.V_move_obstacles, self.H_obstacles, self.H_move_obstacles, self.static_cost,
                                   self.dynamic_cost)

        self.pos_constraints = [0, 20, 0, 60, 0, 20]
        self.simulation_time = 25

    def reset(self):
        self.start = init_obstacles(self.V_obstacles, self.V_move_obstacles, self.H_obstacles, self.H_move_obstacles, self.static_cost,
                                   self.dynamic_cost)


class EnvironmentDynamicLasers:
    def __init__(self, static_cost, dynamic_cost):
        self.pos_constraints = [0, 30, 0, 100, 0, 30]
        self.start = [15, 5, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Initial State: [x, y, z, x_dot, y_dot, z_dot]
        self.goal = [15, 90, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0]



        self.V_obstacles = []
        self.V_move_obstacles = []
        self.H_obstacles = []
        self.H_move_obstacles = []

        ## Define Obstacles
        obstacle_range = [self.pos_constraints[2] + 15, self.pos_constraints[3] - 15]
        dist = obstacle_range[1] - obstacle_range[0]

        radius = 1.6
        x_range = self.pos_constraints[1] - self.pos_constraints[0]
        z_range = self.pos_constraints[5] - self.pos_constraints[4]

        obstacle_count = 20
        for i in range(obstacle_count):
            y_pos = obstacle_range[0] + i * dist / obstacle_count
            if i % 2 == 0:
                self.V_move_obstacles.append([radius/2, y_pos, 3, 0, radius, x_range - radius]) # [center_x, center_y, vel_x, vel_y, radius, move_distance]
            else:
                self.H_move_obstacles.append([y_pos, radius/2, 0, 3, radius, z_range - radius]) # [center_y, center_z, vel_y, vel_z, radius, move_distance]

        self.static_cost = static_cost
        self.dynamic_cost = dynamic_cost

        self.obstacles = init_obstacles(self.V_obstacles, self.V_move_obstacles, self.H_obstacles, self.H_move_obstacles, self.static_cost,
                                   self.dynamic_cost)

        self.simulation_time = 25

    def reset(self):
        self.start = init_obstacles(self.V_obstacles, self.V_move_obstacles, self.H_obstacles, self.H_move_obstacles, self.static_cost,
                                   self.dynamic_cost)


class EnvironmentChimney:
    def __init__(self, static_cost, dynamic_cost):
        self.pos_constraints = [0, 30, 0, 30, 0, 110]
        self.start = [15, 15, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Initial State: [x, y, z, x_dot, y_dot, z_dot]
        self.goal = [15, 15, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.V_obstacles = []
        self.V_move_obstacles = []
        self.H_obstacles = []
        self.H_move_obstacles = []

        ## Define Obstacles
        y_range = [self.pos_constraints[2], self.pos_constraints[3]]
        wall_locations = [25, 45, 65, 85]

        wall_gap = 8

        wall_thickness = 4

        radius_obstacles = 2

        #generate vertical wall with a gap on the left side

        for i in range(len(wall_locations)):
            if i % 2 == 0:
                direction = 1
                start_point = y_range[0] + wall_gap + wall_thickness/2 - wall_thickness
            else:
                direction = -1
                start_point = y_range[1] - wall_gap + wall_thickness/2 + wall_thickness

            y_pos = start_point
            while y_pos < y_range[1] and y_pos > y_range[0]:
                y_pos += wall_thickness * direction
                self.H_obstacles.append([y_pos, wall_locations[i], wall_thickness/2])

            if i != len(wall_locations) - 1:
                move_distance = wall_locations[i+1] - wall_locations[i] - radius_obstacles - wall_thickness
                # [center_y, center_z, vel_y, vel_z, radius, move_distance]
                self.H_move_obstacles.append([15, wall_locations[i] + radius_obstacles/2 + wall_thickness/2, 0, 2, radius_obstacles, move_distance])



        self.static_cost = static_cost
        self.dynamic_cost = dynamic_cost

        self.obstacles = init_obstacles(self.V_obstacles, self.V_move_obstacles, self.H_obstacles, self.H_move_obstacles, self.static_cost,
                                   self.dynamic_cost)

        self.simulation_time = 80

    def reset(self):
        self.start = init_obstacles(self.V_obstacles, self.V_move_obstacles, self.H_obstacles, self.H_move_obstacles, self.static_cost,
                                   self.dynamic_cost)