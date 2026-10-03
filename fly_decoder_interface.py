import numpy as np

from flybrain import FlyBrain


MODEL_FILE = "fly_decoder_model.npz"


class PersistentFlyDecoder:

    def __init__(self):

        print("Loading FLY-001 decoder...")

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

        print(
            f"Decoder loaded."
        )

        print(
            f"Classes: {self.classes}"
        )

        print(
            f"Features: {len(self.features)}"
        )

    def predict(self, X):

        X = np.asarray(
            X,
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

        X_normalized = (
            X / X_norm
        )

        C_normalized = (
            self.centroids / C_norm
        )

        similarities = (
            X_normalized
            @ C_normalized.T
        )

        prediction = np.argmax(
            similarities,
            axis=1
        )

        return [
            self.classes[int(index)]
            for index in prediction
        ]


class FlyBrainInterface:

    def __init__(self):

        print()
        print(
            "Initializing FLY-001 brain..."
        )

        self.brain = FlyBrain(
            device="auto"
        )

        self.azimuth = np.asarray(
            self.brain.azimuth
        )

        self.n_receptors = len(
            self.azimuth
        )

        self.decoder = (
            PersistentFlyDecoder()
        )

        self.positions = {
            "LEFT": -0.75,
            "CENTER_LEFT": -0.375,
            "CENTER": 0.0,
            "CENTER_RIGHT": 0.375,
            "RIGHT": 0.75,
        }

    def make_stimulus(
        self,
        position
    ):

        direction = self.positions[
            position
        ]

        distance = np.abs(
            self.azimuth - direction
        )

        valid = np.where(
            np.isfinite(distance)
        )[0]

        selected = valid[
            np.argsort(
                distance[valid]
            )[:500]
        ]

        stimulus = np.zeros(
            self.n_receptors,
            dtype=np.float32
        )

        stimulus[selected] = 1.0

        return stimulus

    def process(
        self,
        position,
        steps=50
    ):

        self.brain.reset()

        windows = []

        for start in range(
            0,
            steps,
            10
        ):

            window_activity = np.zeros(
                self.brain.n,
                dtype=np.float32
            )

            for _ in range(10):

                stimulus = (
                    self.make_stimulus(
                        position
                    )
                )

                fired = self.brain.step(
                    eye_drive=stimulus
                )

                if len(fired) > 0:

                    window_activity[
                        fired
                    ] += 1

            windows.append(
                window_activity
            )

        neural_state = np.concatenate(
            windows
        )

        decoded = self.decoder.predict(
            neural_state.reshape(1, -1)
        )[0]

        return decoded


def main():

    fly = FlyBrainInterface()

    print()
    print("=" * 60)
    print("FLY-001 NEURAL DECODER TEST")
    print("=" * 60)

    tests = [
        "LEFT",
        "CENTER_LEFT",
        "CENTER",
        "CENTER_RIGHT",
        "RIGHT"
    ]

    correct = 0

    for actual in tests:

        decoded = fly.process(
            actual
        )

        if decoded == actual:
            correct += 1

        print()
        print(
            f"Input neural state : "
            f"{actual}"
        )

        print(
            f"Decoded brain state: "
            f"{decoded}"
        )

    print()
    print(
        f"Accuracy: "
        f"{correct / len(tests) * 100:.2f}%"
    )


if __name__ == "__main__":
    main()