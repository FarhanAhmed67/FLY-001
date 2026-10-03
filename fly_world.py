import math


class FlyWorld:

    def __init__(self, width=100, height=100):

        self.width = width
        self.height = height

        self.fly_x = 50.0
        self.fly_y = 50.0

        self.light_x = 80.0
        self.light_y = 50.0

        self.step_count = 0

    def reset(self):

        self.fly_x = 50.0
        self.fly_y = 50.0

        self.step_count = 0

    def move(self, action):

        step_size = 5.0

        if action == "MOVE_LEFT":
            self.fly_x -= step_size

        elif action == "MOVE_RIGHT":
            self.fly_x += step_size

        elif action == "MOVE_UP":
            self.fly_y += step_size

        elif action == "MOVE_DOWN":
            self.fly_y -= step_size

        elif action == "STAY_CENTER":
            pass

        elif action.startswith("TURN_LEFT"):
            self.fly_x -= step_size

        elif action.startswith("TURN_RIGHT"):
            self.fly_x += step_size

        elif action.startswith("ACTIVE_LEFT"):
            self.fly_x -= step_size * 2

        elif action.startswith("ACTIVE_RIGHT"):
            self.fly_x += step_size * 2

        elif action.startswith("CAUTIOUS_LEFT"):
            self.fly_x -= step_size / 2

        elif action.startswith("CAUTIOUS_RIGHT"):
            self.fly_x += step_size / 2

        self.fly_x = max(
            0.0,
            min(self.width, self.fly_x)
        )

        self.fly_y = max(
            0.0,
            min(self.height, self.fly_y)
        )

        self.step_count += 1

    def get_light_vector(self):

        dx = self.light_x - self.fly_x
        dy = self.light_y - self.fly_y

        distance = math.sqrt(
            dx * dx + dy * dy
        )

        return dx, dy, distance

    def get_light_direction(self):

        dx, dy, distance = (
            self.get_light_vector()
        )

        if abs(dx) < 10:

            return "CENTER"

        if dx < 0:

            return "LEFT"

        return "RIGHT"

    def get_light_vertical(self):

        dx, dy, distance = (
            self.get_light_vector()
        )

        if abs(dy) < 10:

            return "CENTER"

        if dy > 0:

            return "UP"

        return "DOWN"

    def get_state(self):

        dx, dy, distance = (
            self.get_light_vector()
        )

        return {
            "fly_x": self.fly_x,
            "fly_y": self.fly_y,
            "light_x": self.light_x,
            "light_y": self.light_y,
            "distance_to_light": distance,
            "horizontal_direction":
                self.get_light_direction(),
            "vertical_direction":
                self.get_light_vertical(),
            "step": self.step_count
        }

    def describe(self):

        state = self.get_state()

        print()
        print("FLY WORLD")
        print("-" * 50)

        print(
            f"Fly position      : "
            f"({state['fly_x']:.1f}, "
            f"{state['fly_y']:.1f})"
        )

        print(
            f"Light position    : "
            f"({state['light_x']:.1f}, "
            f"{state['light_y']:.1f})"
        )

        print(
            f"Distance to light : "
            f"{state['distance_to_light']:.2f}"
        )

        print(
            f"Horizontal        : "
            f"{state['horizontal_direction']}"
        )

        print(
            f"Vertical          : "
            f"{state['vertical_direction']}"
        )

        print(
            f"World step        : "
            f"{state['step']}"
        )

        print("-" * 50)


if __name__ == "__main__":

    world = FlyWorld()

    print()
    print("INITIAL WORLD")

    world.describe()

    print()
    print("Moving LEFT...")

    world.move("MOVE_LEFT")
    world.describe()

    print()
    print("Moving UP...")

    world.move("MOVE_UP")
    world.describe()

    print()
    print("Moving RIGHT...")

    world.move("MOVE_RIGHT")
    world.describe()

    print()
    print("Moving DOWN...")

    world.move("MOVE_DOWN")
    world.describe()