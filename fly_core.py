import numpy as np
from flybrain import FlyBrain


class Fly001:

    def __init__(self):
        self.brain = FlyBrain(device="auto")
        self.n_visual = len(self.brain.visual)

    def reset(self):
        self.brain.reset()

    def step(self, visual_input):
        """
        Advance the simulated fly by one 20 ms step.
        """

        visual_input = np.asarray(
            visual_input,
            dtype=np.float32
        )

        if len(visual_input) != self.n_visual:
            raise ValueError(
                f"Expected {self.n_visual} visual inputs, "
                f"got {len(visual_input)}"
            )

        fired = self.brain.step(
            eye_drive=visual_input
        )

        return fired

    def get_motor_activity(self, fired):
        """
        Convert fired neuron IDs into motor-group activity.
        """

        fired = np.asarray(fired)

        activity = {}

        for name, neuron_ids in self.brain.groups.items():

            count = 0

            for neuron_id in neuron_ids:
                count += np.sum(fired == neuron_id)

            activity[name] = int(count)

        return activity

    def run(self, visual_input, steps=20, reset=False):
        """
        Run the existing brain state for multiple steps.

        reset=False means the fly keeps its current neural state.
        reset=True starts the fly from a fresh state.
        """

        if reset:
            self.reset()

        total_activity = {
            name: 0
            for name in self.brain.groups
        }

        total_spikes = 0

        for _ in range(steps):

            fired = self.step(visual_input)

            total_spikes += len(fired)

            activity = self.get_motor_activity(fired)

            for name in activity:
                total_activity[name] += activity[name]

        return {
            "total_spikes": total_spikes,
            "motor_activity": total_activity
        }


if __name__ == "__main__":

    fly = Fly001()

    print("FLY-001 initialized")
    print("Visual receptors:", fly.n_visual)
    print("Brain neurons:", fly.brain.n)

    dark = np.zeros(
        fly.n_visual,
        dtype=np.float32
    )

    bright = np.ones(
        fly.n_visual,
        dtype=np.float32
    )

    fly.reset()

    print("\nRunning DARK simulation...")

    dark_state = fly.run(
        dark,
        steps=20
    )

    print("Total spikes:", dark_state["total_spikes"])

    for name, value in dark_state["motor_activity"].items():
        print(f"{name:12} : {value}")

    print("\nRunning BRIGHT simulation...")

    bright_state = fly.run(
        bright,
        steps=20
    )

    print("Total spikes:", bright_state["total_spikes"])

    for name, value in bright_state["motor_activity"].items():
        print(f"{name:12} : {value}")