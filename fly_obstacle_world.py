import math


class FlyObstacleWorld:
    def __init__(self, width=100, height=100):
        self.width = width
        self.height = height

        self.fly_x = 15.0
        self.fly_y = 50.0

        self.target_x = 85.0
        self.target_y = 50.0

        self.obstacle = {
            "x": 45.0,
            "y": 35.0,
            "width": 10.0,
            "height": 30.0,
        }

        self.step_count = 0

    def reset(self):
        self.fly_x = 15.0
        self.fly_y = 50.0
        self.target_x = 85.0
        self.target_y = 50.0
        self.step_count = 0

    def is_inside_obstacle(self, x, y):
        o = self.obstacle

        return (
            o["x"] <= x <= o["x"] + o["width"]
            and
            o["y"] <= y <= o["y"] + o["height"]
        )

    def move(self, dx, dy):
        new_x = self.fly_x + dx
        new_y = self.fly_y + dy

        new_x = max(0.0, min(self.width, new_x))
        new_y = max(0.0, min(self.height, new_y))

        if not self.is_inside_obstacle(new_x, new_y):
            self.fly_x = new_x
            self.fly_y = new_y

        self.step_count += 1

    def get_target_vector(self):
        dx = self.target_x - self.fly_x
        dy = self.target_y - self.fly_y

        distance = math.sqrt(dx * dx + dy * dy)

        return dx, dy, distance

    def get_state(self):
        dx, dy, distance = self.get_target_vector()

        return {
            "fly_x": self.fly_x,
            "fly_y": self.fly_y,
            "target_x": self.target_x,
            "target_y": self.target_y,
            "distance_to_target": distance,
            "target_dx": dx,
            "target_dy": dy,
            "inside_obstacle": self.is_inside_obstacle(
                self.fly_x,
                self.fly_y
            ),
            "step": self.step_count,
        }

    def describe(self):
        state = self.get_state()

        print()
        print("FLY OBSTACLE WORLD")
        print("-" * 50)

        print(
            f"Fly position      : "
            f"({state['fly_x']:.2f}, {state['fly_y']:.2f})"
        )

        print(
            f"Target position   : "
            f"({state['target_x']:.2f}, {state['target_y']:.2f})"
        )

        print(
            f"Distance          : "
            f"{state['distance_to_target']:.2f}"
        )

        print(
            f"Target vector     : "
            f"({state['target_dx']:.2f}, "
            f"{state['target_dy']:.2f})"
        )

        o = self.obstacle

        print(
            f"Obstacle          : "
            f"x={o['x']}, y={o['y']}, "
            f"w={o['width']}, h={o['height']}"
        )

        print(
            f"Collision         : "
            f"{state['inside_obstacle']}"
        )

        print(
            f"World step       : "
            f"{state['step']}"
        )

        print("-" * 50)


if __name__ == "__main__":
    world = FlyObstacleWorld()

    world.describe()

    print("\nTesting movement...")

    world.move(5, 0)
    world.describe()

    world.move(5, 0)
    world.describe()

    print("\nObstacle test complete.")