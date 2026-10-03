import numpy as np

from flybrain import FlyBrain
from fly_behavior import FlyBehavior
from fly_internal_state import FlyInternalState
from fly_world import FlyWorld


MODEL_FILE = "fly_decoder_model.npz"


class PersistentDecoder:

    def __init__(self):

        model = np.load(
            MODEL_FILE,
            allow_pickle=True
        )

        self.features = model["features"]
        self.mean = model["mean"]
        self.std = model["std"]
        self.centroids = model["centroids"]

        self.classes = [
            str(x)
            for x in model["classes"]
        ]

    def predict(self, neural_state):

        X = np.asarray(
            neural_state,
            dtype=np.float32
        )

        X = X[:, self.features]

        X = (
            X - self.mean
        ) / self.std

        X_norm = np.linalg.norm(
            X,
            axis=1,
            keepdims=True
        )

        X_norm[X_norm == 0] = 1.0

        C_norm = np.linalg.norm(
            self.centroids,
            axis=1,
            keepdims=True
        )

        C_norm[C_norm == 0] = 1.0

        X_normalized = X / X_norm

        C_normalized = (
            self.centroids / C_norm
        )

        similarities = (
            X_normalized
            @ C_normalized.T
        )

        index = int(
            np.argmax(
                similarities[0]
            )
        )

        confidence = float(
            similarities[0][index]
        )

        return (
            self.classes[index],
            confidence
        )


class Fly2D:

    def __init__(self, brain=None):

        self.world = FlyWorld()

        self.brain = (
            brain
            if brain is not None
            else FlyBrain(device="auto")
        )

        self.decoder = (
            PersistentDecoder()
        )

        self.behavior = FlyBehavior()

        self.internal_state = (
            FlyInternalState()
        )

        self.azimuth = np.asarray(
            self.brain.azimuth
        )

        self.n_receptors = len(
            self.azimuth
        )

    def make_visual_input(self):

        direction = (
            self.world.get_light_direction()
        )

        direction_map = {
            "LEFT": -0.75,
            "CENTER": 0.0,
            "RIGHT": 0.75
        }

        target = direction_map[
            direction
        ]

        distance = np.abs(
            self.azimuth - target
        )

        valid = np.where(
            np.isfinite(distance)
        )[0]

        selected = valid[
            np.argsort(
                distance[valid]
            )[:500]
        ]

        visual = np.zeros(
            self.n_receptors,
            dtype=np.float32
        )

        visual[selected] = 1.0

        return visual

    def process_brain(self):

        self.brain.reset()

        visual = (
            self.make_visual_input()
        )

        total_spikes = 0

        windows = []

        for _ in range(5):

            window = np.zeros(
                self.brain.n,
                dtype=np.float32
            )

            for _ in range(10):

                fired = self.brain.step(
                    eye_drive=visual
                )

                total_spikes += len(
                    fired
                )

                if len(fired) > 0:

                    window[fired] += 1

            windows.append(window)

        neural_state = np.concatenate(
            windows
        )

        decoded, confidence = (
            self.decoder.predict(
                neural_state.reshape(
                    1,
                    -1
                )
            )
        )

        return (
            decoded,
            confidence,
            total_spikes
        )

    def update_internal_state(
        self,
        decoded,
        spikes
    ):

        previous = (
            self.internal_state
            .current_direction
        )

        if previous is None:

            previous = decoded

        summary = {

            "current_state": {
                "direction": decoded
            },

            "previous_state": {
                "direction": previous
            },

            "direction_changes":
                0
                if decoded == previous
                else 1,

            "pattern":
                "STABLE"
                if decoded == previous
                else "CHANGING",

            "interactions":
                self.world.step_count + 1,

            "recent_directions":
                [decoded]
        }

        self.internal_state.update(
            summary,
            spikes
        )

    def convert_behavior_to_world_action(
        self,
        action
    ):

        if "LEFT" in action:

            return "MOVE_LEFT"

        if "RIGHT" in action:

            return "MOVE_RIGHT"

        if "CENTER" in action:

            return "STAY_CENTER"

        return "STAY_CENTER"

    def run_step(self):

        (
            decoded,
            confidence,
            spikes
        ) = self.process_brain()

        self.update_internal_state(
            decoded,
            spikes
        )

        action = self.behavior.decide(
            decoded,
            self.internal_state
        )

        world_action = (
            self.convert_behavior_to_world_action(
                action
            )
        )

        state_before = (
            self.world.get_state()
        )

        self.world.move(
            world_action
        )

        state_after = (
            self.world.get_state()
        )

        print()
        print("=" * 60)

        print(
            f"STEP {self.world.step_count}"
        )

        print("=" * 60)

        print(
            f"Fly position      : "
            f"({state_before['fly_x']:.1f}, "
            f"{state_before['fly_y']:.1f})"
        )

        print(
            f"Light position    : "
            f"({state_before['light_x']:.1f}, "
            f"{state_before['light_y']:.1f})"
        )

        print(
            f"Light direction   : "
            f"{state_before['horizontal_direction']}"
        )

        print(
            f"Brain decoded     : "
            f"{decoded}"
        )

        print(
            f"Similarity        : "
            f"{confidence:.4f}"
        )

        print(
            f"Neural spikes     : "
            f"{spikes}"
        )

        print(
            f"Behavior          : "
            f"{action}"
        )

        print(
            f"World action      : "
            f"{world_action}"
        )

        print(
            f"New position      : "
            f"({state_after['fly_x']:.1f}, "
            f"{state_after['fly_y']:.1f})"
        )

        print(
            f"New distance      : "
            f"{state_after['distance_to_light']:.2f}"
        )

        print(
            f"New light relation: "
            f"{state_after['horizontal_direction']}"
        )


if __name__ == "__main__":

    simulation = Fly2D()

    print()
    print("=" * 60)
    print("FLY-001 2D BRAIN CONTROL TEST")
    print("=" * 60)

    print()
    print(
        "Fly starts at (50,50)"
    )

    print(
        "Light starts at (80,50)"
    )

    print()
    print(
        "Running 10 brain-controlled steps..."
    )

    for _ in range(10):

        simulation.run_step()