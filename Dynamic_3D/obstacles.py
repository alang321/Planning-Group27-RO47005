class CylinderVertical:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

    def is_colliding(self, point):
        squared_dist = (point[0] - self.x) ** 2 + (point[1] - self.y) ** 2
        if squared_dist <= self.radius ** 2:
            return True
        return False

    def


class CylinderHorizontal:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

    def is_colliding(self, point):
        squared_dist = (point[0] - self.x) ** 2 + (point[1] - self.y) ** 2
        if squared_dist <= self.radius ** 2:
            return True
        return False