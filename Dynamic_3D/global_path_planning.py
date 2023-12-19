from ipywidgets import IntProgress
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import operator

from mpl_toolkits.mplot3d import art3d


def rrt_star(world_3d, start, goal, radius, max_iter=1000, report_progress=True):
    # Initialize start and goal node
    start_node = Node(None, start)
    goal_node = Node(None, goal)
    start_node.g = 0

    nodes = [start_node]

    # squared for faster nearest node search
    radius_sq = radius ** 2

    success = False

    if report_progress:
        f = IntProgress(min=0, max=max_iter)  # instantiate the bar
        display(f)  # display the bar

    for i in range(max_iter):
        if report_progress:
            if i % 10 == 0:
                f.value = i
            if i % 1000 == 0:
                print(f"Success state: {success}")

        rnd_pos = world_3d.get_random_point()

        if world_3d.is_colliding(rnd_pos):
            continue

        # Find nearest valid node
        nearest_node = None
        nearest_dist_sq = None
        dist_sq = [(node.position[0] - rnd_pos[0]) ** 2 + (node.position[1] - rnd_pos[1]) ** 2 for node in nodes]

        sorted_nodes = [(x, dist) for dist, x in sorted(zip(dist_sq, nodes), key=lambda pair: pair[0])]
        for node, dist in sorted_nodes:
            if not world_3d.is_line_colliding(*node.position, *rnd_pos):
                nearest_node = node
                nearest_dist_sq = dist

        if nearest_node is None:
            continue

        new_node = Node(nearest_node, rnd_pos)
        new_node.g = nearest_dist_sq ** 0.5 + nearest_node.g

        # check if nearby nodes are close to the new node
        for idx, node in enumerate(nodes):
            dist_sq = node.distance_sq(new_node)
            if dist_sq > radius_sq:
                continue

            # check if line is colliding
            if world_3d.is_line_colliding(*node.position, *new_node.position):
                continue

            # check if new node is closer path to the node
            dist = dist_sq ** 0.5
            if new_node.g + dist < node.g:
                nodes[idx].parent = new_node
                nodes[idx].g = new_node.g + dist

        nodes.append(new_node)

        dist_sq = new_node.distance_sq(goal_node)
        if dist_sq < 2 * radius_sq:
            if not world_3d.is_line_colliding(*new_node.position, *goal_node.position):
                dist = dist_sq ** 0.5
                if goal_node.g is None or new_node.g + dist < goal_node.g:
                    goal_node.parent = new_node
                    goal_node.g = new_node.g + dist

                success = True

    if success:
        print("Goal found")
        #reconstruct the path
        path = []
        current = goal_node
        i = 0
        while current is not None:
            path.append(current.position)
            current = current.parent
            i += 1
            if i > max_iter:
                print("Infinite loop")
                return Path(None, start, goal, "RRT*", world_3d, False)
        return Path(path[::-1], start, goal, "RRT*", world_3d, success)
    else:
        print("No path found")
        return Path(None, start, goal, "RRT*", world_3d, success)




class World_3D:
    def __init__(self, x_range, y_range, z_range, obstacles, obstacle_margin=0):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.obstacles = obstacles
        self.obstacle_margin = obstacle_margin

    def is_colliding(self, point, margin=0):
        for obstacle in self.obstacles:
            squared_dist = (point[0] - obstacle[0]) ** 2 + (point[1] - obstacle[1]) ** 2
            if squared_dist <= (obstacle[2] + margin + self.obstacle_margin) ** 2:
                return True
        return False

    def is_line_colliding(self, x0, y0, x1, y1, margin=0, point_spacing=0.1):
        #create points along the line with given spacing
        dist = ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
        point_fractions = np.arange(0, 1, point_spacing)

        for fraction in point_fractions:
            x = x0 + fraction * (x1 - x0)
            y = y0 + fraction * (y1 - y0)
            if self.is_colliding((x, y), margin):
                return True

        if self.is_colliding((x1, y1), margin):
            return True

        return False

    def get_random_point(self):
        x = np.random.uniform(self.x_range[0], self.x_range[1])
        y = np.random.uniform(self.y_range[0], self.y_range[1])
        z = np.random.uniform(self.z_range[0], self.z_range[1])
        return x, y, z

    def is_in_constraints(self, pos_x, pos_y):
        return pos_x >= self.x_range[0] and pos_x <= self.x_range[1] and pos_y >= self.y_range[0] and pos_y <= self.y_range[1]


    def plot(self, path=None):
        #plot the 3d cylinder obstacles and the path from the top side and front
        # x = 0, y = 1, z = 2
        #for direction in range(3):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        #plot obstacles
        for obstacle in self.obstacles:
            self.plot_cylinder_xy(ax, obstacle[0], obstacle[1], obstacle[2], 'red')

        if path is not None:
            path_points = np.array(path.path_points)
            ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], color='blue')
            #plot start and end point
            ax.scatter(path.start[0], path.start[1], path.start[2], color='green')
            ax.scatter(path.goal[0], path.goal[1], path.goal[2], color='green')
        
        # Set the origin (0, 0, 0) point at the center of the plot
        ax.plot([0], [0], [0], marker='o', markersize=5, color='black')  # Plot a point at the origin

        ax.set_aspect('equal', adjustable='box')

        ax.set_xlim(self.x_range)
        ax.set_ylim(self.y_range)
        ax.set_zlim(self.z_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def plot_circle_xy(self, ax, x, y, z, radius, color):
        #this function plots only a circle in 3d
        theta = np.linspace(0, 2 * np.pi, 201)
        x_circ = x + radius * np.cos(theta)
        y_circ = y + radius * np.sin(theta)

        ax.plot(x_circ, y_circ, z, color=color)

        return ax

    def plot_cylinder_xy(self, ax, x, y, radius, color):
        #plot a cylinder from lines and circles in 3d
        #plot the circles
        self.plot_circle_xy(ax, x, y, self.z_range[0], radius, color)
        self.plot_circle_xy(ax, x, y, self.z_range[1], radius, color)

        #plot the center line
        #ax.plot([x, x], [y, y], self.world_3d.z_range)
        ax.plot([x, x], [y + radius, y + radius], self.z_range, color=color)
        ax.plot([x, x], [y - radius, y - radius], self.z_range, color=color)
        ax.plot([x + radius, x + radius], [y, y], self.z_range, color=color)
        ax.plot([x - radius, x - radius], [y, y], self.z_range, color=color)


class Node:
    def __init__(self, parent, position):
        self.parent = parent
        self.position = position

        self.g = None # Distance to start node
        self.h = None # Heuristic
        self.f = None # Total cost, distance to start + heuristic

    def distance(self, other):
        return (self.position[0] - other.position[0]) ** 2 + (self.position[1] - other.position[1]) ** 2

    def distance_sq(self, other):
        return (self.position[0] - other.position[0]) ** 2 + (self.position[1] - other.position[1]) ** 2

    def __eq__(self, other):
        return self.position == other.position

    def __ne__(self, other):
        return self.position != other.position

    def __lt__(self, other):
        return self.f < other.f

    def __le__(self, other):
        return self.f <= other.f

    def __gt__(self, other):
        return self.f > other.f

    def __ge__(self, other):
        return self.f >= other.f

    def __str__(self):
        return f'Node: {self.position}, g: {self.g}'

class Path:
    def __init__(self, path_points, start, goal, name, world_3d, valid=True):
        self.name = name
        self.start = start
        self.goal = goal
        self.valid = valid
        self.world_3d = world_3d

        self.path_points = path_points





            
