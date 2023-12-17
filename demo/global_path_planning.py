
from ipywidgets import IntProgress
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import operator

def rrt_star(occupany_grid, start, goal, radius, max_iter=1000, report_progress=True):
    # Initialize start and goal node
    start_idx = occupany_grid.get_closest_grid_index(start)
    goal_idx = occupany_grid.get_closest_grid_index(goal)
    start_node = Node(None, start_idx)
    goal_node = Node(None, goal_idx)
    start_node.g = 0

    visited_points = np.zeros(occupany_grid.grid_map.shape)

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

        rnd_idx = occupany_grid.get_random_cell_idx()

        #is the random point already visitedä
        if visited_points[rnd_idx] == 1:
            continue

        visited_points[rnd_idx] = 1

        if occupany_grid.is_occupied_index(*rnd_idx):
            continue

        # Find nearest valid node
        nearest_node = None
        nearest_dist_sq = None
        dist_sq = [(node.index[0] - rnd_idx[0]) ** 2 + (node.index[1] - rnd_idx[1]) ** 2 for node in nodes]

        print(rnd_idx)
        print(dist_sq)
        sorted_nodes = [(x, dist) for dist, x in sorted(zip(dist_sq, nodes), key=lambda pair: pair[0])]
        for i in nodes:
            print(i.index)
        print("end")
        print(sorted_nodes[0][0].index, sorted_nodes[0][1])
        for node, dist in sorted_nodes:
            if not occupany_grid.is_line_intersecting_obstacle(*node.index, *rnd_idx):
                nearest_node = node
                nearest_dist_sq = dist

        if nearest_node is None:
            print()
            continue


        print(nearest_node.index)
        print()


        new_node = Node(nearest_node, rnd_idx)
        new_node.g = nearest_dist_sq ** 0.5 + nearest_node.g

        # check if nearby nodes are close to the new node
        for idx, node in enumerate(nodes):
            dist_sq = node.distance_sq(new_node)
            if dist_sq > radius_sq:
                continue

            # check if new node is closer to the node
            if occupany_grid.is_line_intersecting_obstacle(*node.index, *new_node.index):
                continue

            dist = dist_sq ** 0.5
            if new_node.g + dist < node.g:
                node.parent = new_node
                node.g = new_node.g + dist

        nodes.append(new_node)

        dist_sq = new_node.distance_sq(goal_node)
        if dist_sq < 2 * radius_sq:
            if not occupany_grid.is_line_intersecting_obstacle(*new_node.index, *goal_node.index):
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
            path.append(current.index)
            current = current.parent
            i += 1
            if i > max_iter:
                print("Infinite loop")
                return Path(None, start, goal, "RRT*", occupany_grid, False)
        return Path(path[::-1], start, goal, "RRT*", occupany_grid, success)
    else:
        print("No path found")
        return Path(None, start, goal, "RRT*", occupany_grid, success)


def a_star(occupany_grid, start, goal, report_progress=True):
    def heuristic(node1, node2):
        return (node1.index[0] - node2.index[0]) ** 2 + (node1.index[1] - node2.index[1]) ** 2

    #lists for open and closed nodes
    open_nodes = []
    closed_nodes = []

    # Initialize start and goal node
    start_idx = occupany_grid.get_closest_grid_index(start)
    goal_idx = occupany_grid.get_closest_grid_index(goal)
    start_node = Node(None, start_idx)
    goal_node = Node(None, goal_idx)
    start_node.g = 0
    start_node.h = 0
    start_node.f = 0
    goal_node.g = 0
    goal_node.h = 0
    goal_node.f = 0

    open_nodes.append(start_node)

    nodes_visited = 0
    if report_progress:
        total_nodes = occupany_grid.num_free_cells
        f = IntProgress(min=0, max=total_nodes)  # instantiate the bar
        display(f)  # display the bar
    # Loop until goal is found
    while len(open_nodes) > 0:
        if(report_progress):
            f.value = nodes_visited

        # Get the cheapest open node measured by f value, f value is defined in the node class operator overloading
        min_idx, cheapest_open_node = min(enumerate(open_nodes), key=operator.itemgetter(1))

        # Move from open to closed list
        closed_nodes.append(open_nodes.pop(min_idx))

        # Check if goal is found
        if cheapest_open_node == goal_node:
            print("Goal found")
            #reconstruct the path
            path = []
            current = cheapest_open_node
            while current is not None:
                path.append(current.index)
                current = current.parent
            return Path(path[::-1], start, goal, "A*", occupany_grid, True)

        # Generate children
        for neighbor_idx, cost in occupany_grid.get_adjacent_free_cells(*cheapest_open_node.index):
            # Create neighbor node
            neighbor_node = Node(cheapest_open_node, neighbor_idx)

            # Check if neighbor is in closed list
            if not neighbor_node in closed_nodes:
                # Calculate f, g, and h values
                neighbor_node.g = cheapest_open_node.g + cost
                neighbor_node.h = heuristic(neighbor_node, cheapest_open_node)
                neighbor_node.f = neighbor_node.g + neighbor_node.h

                #check if neighbor is in open list
                #if not a value error is thrown
                try:
                    index = open_nodes.index(neighbor_node)
                    if open_nodes[index].g > neighbor_node.g:
                        # if its cheaper to get to the current neighbor with the current path
                        # update the open node, otherwise discard it
                        open_nodes[index] = neighbor_node

                except ValueError:
                    # if the neighbor is not in the open list add it
                    open_nodes.append(neighbor_node)
                    nodes_visited += 1
                    pass

    return Path(None, start, goal, "A*", occupany_grid, False)




class OccupancyGrid:
    def __init__(self, grid_resolution, origin, obstacles, field_size, obstacle_margin=0):
        self.grid_resolution = grid_resolution
        self.origin = origin
        self.obstacles = obstacles
        self.field_size = field_size
        self.obstacle_margin = obstacle_margin

        self.x_points = np.arange(self.origin[0], self.field_size[0], self.grid_resolution)
        self.y_points = np.arange(self.origin[1], self.field_size[1], self.grid_resolution)


        self.diag_cost = (2 ** 0.5) * self.grid_resolution
        self.straight_cost = self.grid_resolution

        self.num_free_cells = 0
        self.grid_map = self.generate_grid_map()


    def get_closest_grid_index(self, point):
        x_points = np.arange(self.origin[0], self.field_size[0], self.grid_resolution)
        y_points = np.arange(self.origin[1], self.field_size[1], self.grid_resolution)
        x_index = np.argmin(np.abs(x_points - point[0]))
        y_index = np.argmin(np.abs(y_points - point[1]))
        return x_index, y_index

    def index_to_point(self, index_x, index_y):
        x = self.x_points[index_x]
        y = self.y_points[index_y]
        return [x, y]

    def generate_grid_map(self):
        x_dim = self.x_points.size
        y_dim = self.y_points.size
        grid_map = np.zeros((x_dim, y_dim))
        for i, x in enumerate(self.x_points):
            for j, y in enumerate(self.y_points):
                if self.is_colliding([x,y], self.grid_resolution / 2):
                    grid_map[i, j] = 1
                else:
                    self.num_free_cells += 1
        return grid_map

    def is_colliding(self, point, radius):
        for obstacle in self.obstacles:
            squared_dist = (point[0] - obstacle[0]) ** 2 + (point[1] - obstacle[1]) ** 2
            if squared_dist <= (obstacle[2] + radius + self.obstacle_margin) ** 2:
                return True
        return False

    def is_line_intersecting_obstacle(self, x0, y0, x1, y1):
        # Bresenham's line algorithm
        #https://www.javatpoint.com/computer-graphics-bresenhams-line-algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        x, y = x0, y0

        if x0 < x1:
            x_inc = 1
        else:
            x_inc = -1

        if y0 < y1:
            y_inc = 1
        else:
            y_inc = -1

        p = 2 * dy - dx


        plt.imshow(self.grid_map.T, cmap='Greys', origin='lower')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Occupancy Grid')

        while x != x1:
            plt.scatter(x, y, color = "r")
            if self.is_occupied_index(x, y):
                return True

            if p >= 0:
                y += y_inc
                p += 2 * dy - 2 * dx
            else:
                p += 2 * dy

            x += x_inc


        plt.show()

        return False

    def get_random_cell_idx(self):
        index_x = np.random.randint(0, len(self.grid_map[0]))
        index_y = np.random.randint(0, len(self.grid_map[1]))
        return index_x, index_y

    def get_adjacent_free_cells(self, index_x, index_y):
        adjacent_indices = []
        costs = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == j == 0:
                    continue
                new_index_x = index_x + i
                new_index_y = index_y + j
                if self.is_free_valid_cell(new_index_x, new_index_y):
                    #check if diagonal or straight
                    if i == 0 or j == 0:
                        cost = self.straight_cost
                    else:
                        cost = self.diag_cost
                    adjacent_indices.append(((new_index_x, new_index_y), cost))
        return adjacent_indices

    def is_free_valid_cell(self, index_x, index_y):
        return index_x >= 0 and index_x < len(self.grid_map[0]) and index_y >= 0 and index_y < len(self.grid_map[1]) and self.grid_map[index_x, index_y] != 1

    def is_occupied_index(self, index_x, index_y):
        return self.grid_map[index_x, index_y] == 1

    def plot(self):
        #flip the axes to match the grid
        plt.imshow(self.grid_map.T, cmap='Greys', origin='lower')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Occupancy Grid')
        plt.show()


class Node:
    def __init__(self, parent, index):
        self.parent = parent
        self.index = index

        self.g = None # Distance to start node
        self.h = None # Heuristic
        self.f = None # Total cost, distance to start + heuristic

    def distance(self, other):
        return (self.index[0] - other.index[0]) ** 2 + (self.index[1] - other.index[1]) ** 2

    def distance_sq(self, other):
        return (self.index[0] - other.index[0]) ** 2 + (self.index[1] - other.index[1]) ** 2

    def __eq__(self, other):
        return self.index == other.index

    def __ne__(self, other):
        return self.index != other.index

    def __lt__(self, other):
        return self.f < other.f

    def __le__(self, other):
        return self.f <= other.f

    def __gt__(self, other):
        return self.f > other.f

    def __ge__(self, other):
        return self.f >= other.f

    def __str__(self):
        return f'Node: {self.index}, g: {self.g}'

class Path:
    def __init__(self, path_indeces, start, point, name, occupany_grid, valid=True):
        self.path_indeces = path_indeces
        self.name = name
        self.start = start
        self.point = point
        self.valid = valid
        self.occupany_grid = occupany_grid

        if self.valid:
            # convert indeces to points
            path_points = []
            for index in path_indeces:
                path_points.append(occupany_grid.index_to_point(*index))
            self.path_points = np.array(path_points)

    def plot(self):
        if self.valid:
            pos_x = self.path_points[:, 0]
            pos_y = self.path_points[:, 1]
            plt.plot(pos_x, pos_y, color = "r", label = "Vehicle Path")

        plt.title(self.name)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        #mark start and goal
        plt.scatter(self.start[0], self.start[1], color = "g", label = "Start")
        plt.scatter(self.point[0], self.point[1], color = "pink", label = "Goal")
        plt.legend()

        #plot obstacles
        for idx, obstacle in enumerate(self.occupany_grid.obstacles):
            circle = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='blue', fill=False)
            plt.gca().add_patch(circle)

        # Set axis limits for x and y
        plt.xlim(0, 50)
        plt.ylim(0, 50)
        plt.axis('equal')
        plt.grid()
        plt.show()