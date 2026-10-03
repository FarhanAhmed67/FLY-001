import numpy as np
import math

from flybrain import FlyBrain
from fly_behavior import FlyBehavior
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
        C_normalized = self.centroids / C_norm

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


class ContinuousFly:

    def __init__(self):

        self.world = FlyWorld()

        self.brain = FlyBrain(
            device="auto"
        )

        self.decoder = (
            PersistentDecoder()
        )

        self.behavior = FlyBehavior()

        self.azimuth = np.asarray(
            self.brain.azimuth
        )

        self.n_receptors = len(
            self.azimuth
        )

    def get_continuous_visual_target(self):

        dx = (
            self.world.light_x
            - self.world.fly_x
        )

        dy = (
            self.world.light_y
            - self.world.fly_y
        )

        distance = math.sqrt(
            dx * dx + dy * dy
        )

        if distance == 0:

            return 0.0

        normalized = dx / 40.0

        return float(
            np.clip(
                normalized,
                -1.0,
                1.0
            )
        )

    def make_visual_input(self):

        target = (
            self.get_continuous_visual_target()
        )

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

        windows = []

        total_spikes = 0

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

            windows.append(
                window
            )

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

    def action_from_decoded(
        self,
        decoded
    ):

        if decoded == "LEFT":

            return "MOVE_LEFT"

        if decoded == "RIGHT":

            return "MOVE_RIGHT"

        return "STAY_CENTER"

    def run_step(self):

        target = (
            self.get_continuous_visual_target()
        )

        (
            decoded,
            confidence,
            spikes
        ) = self.process_brain()

        action = (
            self.action_from_decoded(
                decoded
            )
        )

        before_x = (
            self.world.fly_x
        )

        before_y = (
            self.world.fly_y
        )

        before_distance = (
            self.world.get_state()[
                "distance_to_light"
            ]
        )

        self.world.move(
            action
        )

        after_distance = (
            self.world.get_state()[
                "distance_to_light"
            ]
        )

        print()
        print("=" * 60)

        print(
            f"STEP {self.world.step_count}"
        )

        print("=" * 60)

        print(
            f"Fly position      : "
            f"({before_x:.1f}, "
            f"{before_y:.1f})"
        )

        print(
            f"Light position    : "
            f"({self.world.light_x:.1f}, "
            f"{self.world.light_y:.1f})"
        )

        print(
            f"Continuous visual : "
            f"{target:+.3f}"
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
            f"Action            : "
            f"{action}"
        )

        print(
            f"New position      : "
            f"({self.world.fly_x:.1f}, "
            f"{self.world.fly_y:.1f})"
        )

        print(
            f"Distance before   : "
            f"{before_distance:.2f}"
        )

        print(
            f"Distance after    : "
            f"{after_distance:.2f}"
        )


if __name__ == "__main__":

    simulation = ContinuousFly()

    print()
    print("=" * 60)
    print("FLY-001 CONTINUOUS VISUAL EXPERIMENT")
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
        "The brain receives a continuous"
    )

    print(
        "visual position rather than only"
    )

    print(
        "LEFT / CENTER / RIGHT."
    )

    print()
    print(
        "Running 10 steps..."
    )

    for _ in range(10):

        simulation.run_step()