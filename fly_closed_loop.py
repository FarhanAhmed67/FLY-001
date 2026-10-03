from fly_world import FlyWorld
from fly_behavior import FlyBehavior


class FlyClosedLoop:

    def __init__(self):

        self.world = FlyWorld()
        self.behavior = FlyBehavior()

    def run_step(self, action):

        print()
        print("=" * 50)
        print("CLOSED LOOP STEP")
        print("=" * 50)

        print()
        print("BEFORE ACTION")

        self.world.describe()

        print()
        print(
            f"Behavior action: {action}"
        )

        self.world.move(action)

        print()
        print("AFTER ACTION")

        self.world.describe()


if __name__ == "__main__":

    simulation = FlyClosedLoop()

    simulation.run_step(
        "MOVE_LEFT"
    )

    simulation.run_step(
        "MOVE_LEFT"
    )

    simulation.run_step(
        "MOVE_RIGHT"
    )