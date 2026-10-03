import numpy as np

from flybrain import FlyBrain
from fly_behavior import FlyBehavior
from fly_internal_state import FlyInternalState


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

        X_norm[
            X_norm == 0
        ] = 1.0

        C_norm = np.linalg.norm(
            self.centroids,
            axis=1,
            keepdims=True
        )

        C_norm[
            C_norm == 0
        ] = 1.0

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


class ClosedLoopWorld:

    def __init__(self):

        self.width = 100

        self.fly_x = 50.0

        self.light_x = 80.0

        self.step_count = 0

        self.brain = FlyBrain(
            device="auto"
        )

        self.decoder = (
            PersistentDecoder()
        )

        self.internal_state = (
            FlyInternalState()
        )

        self.behavior = FlyBehavior()

        self.azimuth = np.asarray(
            self.brain.azimuth
        )

        self.n_receptors = len(
            self.azimuth
        )

    def get_light_direction(self):

        difference = (
            self.light_x - self.fly_x
        )

        if difference < -10:

            return "LEFT"

        elif difference > 10:

            return "RIGHT"

        return "CENTER"

    def make_visual_input(self):

        direction = (
            self.get_light_direction()
        )

        directions = {
            "LEFT": -0.75,
            "CENTER": 0.0,
            "RIGHT": 0.75
        }

        target = directions[
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

        visual_input = (
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
                    eye_drive=visual_input
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

    def update_internal_state(
        self,
        neural_activity,
        decoded
    ):

        previous = (
            self.internal_state
            .current_direction
        )

        if previous is None:

            previous = decoded

        memory_summary = {

            "current_state": {
                "direction": decoded
            },

            "previous_state": {
                "direction": previous
            },

            "direction_changes":
                0 if previous == decoded else 1,

            "pattern":
                "STABLE"
                if previous == decoded
                else "CHANGING",

            "interactions":
                self.step_count + 1,

            "recent_directions":
                [decoded]
        }

        self.internal_state.update(
            memory_summary,
            neural_activity
        )

    def apply_action(
        self,
        action
    ):

        step_size = 5.0

        if action == "MOVE_LEFT":

            self.fly_x -= step_size

        elif action == "MOVE_RIGHT":

            self.fly_x += step_size

        elif action == "STAY_CENTER":

            pass

        elif action.startswith(
            "TURN_LEFT"
        ):

            self.fly_x -= step_size

        elif action.startswith(
            "TURN_RIGHT"
        ):

            self.fly_x += step_size

        elif action.startswith(
            "ACTIVE_LEFT"
        ):

            self.fly_x -= step_size * 2

        elif action.startswith(
            "ACTIVE_RIGHT"
        ):

            self.fly_x += step_size * 2

        elif action.startswith(
            "CAUTIOUS_LEFT"
        ):

            self.fly_x -= step_size / 2

        elif action.startswith(
            "CAUTIOUS_RIGHT"
        ):

            self.fly_x += step_size / 2

        self.fly_x = max(
            0.0,
            min(
                self.width,
                self.fly_x
            )
        )

        self.step_count += 1

    def run_step(self):

        direction_before = (
            self.get_light_direction()
        )

        decoded, confidence, spikes = (
            self.process_brain()
        )

        self.update_internal_state(
            spikes,
            decoded
        )

        action = self.behavior.decide(
            decoded,
            self.internal_state
        )

        print()
        print("=" * 60)

        print(
            f"STEP {self.step_count + 1}"
        )

        print("=" * 60)

        print(
            f"Fly position       : "
            f"{self.fly_x:.1f}"
        )

        print(
            f"Light position     : "
            f"{self.light_x:.1f}"
        )

        print(
            f"Light seen as      : "
            f"{direction_before}"
        )

        print(
            f"Brain decoded      : "
            f"{decoded}"
        )

        print(
            f"Decoder similarity : "
            f"{confidence:.4f}"
        )

        print(
            f"Neural spikes      : "
            f"{spikes}"
        )

        print(
            f"Internal drive     : "
            f"{self.internal_state.get_internal_drive():.3f}"
        )

        print(
            f"Behavior           : "
            f"{action}"
        )

        self.apply_action(
            action
        )

        print(
            f"New fly position   : "
            f"{self.fly_x:.1f}"
        )

        print(
            f"New light relation : "
            f"{self.get_light_direction()}"
        )


if __name__ == "__main__":

    simulation = ClosedLoopWorld()

    print()
    print("=" * 60)
    print("FLY-001 BRAIN → BEHAVIOR → WORLD")
    print("=" * 60)

    print()
    print(
        "Light starts at X=80"
    )

    print(
        "Fly starts at X=50"
    )

    print(
        "Running 10 closed-loop steps..."
    )

    for _ in range(10):

        simulation.run_step()