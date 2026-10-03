import numpy as np

from flybrain import FlyBrain
from fly_decoder import FlyDecoder


POSITIONS = {
    "LEFT": -0.75,
    "CENTER_LEFT": -0.375,
    "CENTER": 0.0,
    "CENTER_RIGHT": 0.375,
    "RIGHT": 0.75,
}


MODEL_FILE = "fly_decoder_model.npz"


class FlyBrainDecoder:

    def __init__(
        self,
        trials_per_class=10,
        steps=50
    ):

        print("Initializing brain decoder...")

        self.brain = FlyBrain(
            device="auto"
        )

        self.azimuth = np.asarray(
            self.brain.azimuth
        )

        self.steps = steps

        self.n_receptors = len(
            self.azimuth
        )

        self.classes = list(
            POSITIONS.keys()
        )

        self.decoder = FlyDecoder(
            classes=list(
                range(len(self.classes))
            ),
            top_features=1000
        )

        print(
            f"Neurons: {self.brain.n}"
        )

        print(
            f"Receptors: {self.n_receptors}"
        )

        print(
            f"Classes: {self.classes}"
        )

        self.train(
            trials_per_class
        )

    def make_stimulus(
        self,
        position,
        intensity=1.0
    ):

        direction = POSITIONS[
            position
        ]

        distance = np.abs(
            self.azimuth - direction
        )

        valid = np.where(
            np.isfinite(distance)
        )[0]

        selected = valid[
            np.argsort(distance[valid])[:500]
        ]

        stimulus = np.zeros(
            self.n_receptors,
            dtype=np.float32
        )

        stimulus[selected] = intensity

        return stimulus

    def collect_sample(
        self,
        position
    ):

        self.brain.reset()

        windows = []

        for start in range(
            0,
            self.steps,
            10
        ):

            window_activity = np.zeros(
                self.brain.n,
                dtype=np.float32
            )

            for _ in range(10):

                stimulus = self.make_stimulus(
                    position
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

        return np.concatenate(
            windows
        )

    def train(
        self,
        trials_per_class
    ):

        print()
        print("=" * 60)
        print("TRAINING BRAIN DECODER")
        print("=" * 60)

        X = []
        y = []

        for class_index, position in enumerate(
            self.classes
        ):

            for trial in range(
                trials_per_class
            ):

                print(
                    f"Training "
                    f"{position:12s} "
                    f"trial "
                    f"{trial + 1}/"
                    f"{trials_per_class}"
                )

                sample = self.collect_sample(
                    position
                )

                X.append(sample)
                y.append(class_index)

        X = np.asarray(
            X,
            dtype=np.float32
        )

        y = np.asarray(y)

        print()
        print(
            "Training matrix:",
            X.shape
        )

        self.decoder.fit(
            X,
            y
        )

        print(
            "Brain decoder trained."
        )

    def save(self):

        np.savez(
            MODEL_FILE,
            features=self.decoder.features,
            mean=self.decoder.mean,
            std=self.decoder.std,
            centroids=self.decoder.centroids,
            classes=np.asarray(
                self.classes
            )
        )

        print()
        print(
            f"Decoder saved to {MODEL_FILE}"
        )

    def decode(
        self,
        position
    ):

        sample = self.collect_sample(
            position
        )

        prediction = self.decoder.predict(
            sample.reshape(1, -1)
        )

        index = int(
            prediction[0]
        )

        return self.classes[index]


def main():

    decoder = FlyBrainDecoder(
        trials_per_class=10
    )

    decoder.save()

    print()
    print("=" * 60)
    print("DECODER TEST")
    print("=" * 60)

    correct = 0
    total = 0

    for position in decoder.classes:

        predicted = decoder.decode(
            position
        )

        total += 1

        if predicted == position:
            correct += 1

        print(
            f"Actual: {position:12s} "
            f"Decoded: {predicted}"
        )

    print()

    print(
        f"Accuracy: "
        f"{correct / total * 100:.2f}%"
    )

    print(
        f"Chance: "
        f"{100 / len(decoder.classes):.2f}%"
    )


if __name__ == "__main__":
    main()