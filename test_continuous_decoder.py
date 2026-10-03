import numpy as np

from flybrain import FlyBrain


MODEL_FILE = "fly_continuous_decoder.npz"


TEST_POSITIONS = np.arange(
    -0.95,
    1.0,
    0.10
)

TRIALS_PER_POSITION = 2


class ContinuousDecoder:

    def __init__(self):

        model = np.load(
            MODEL_FILE,
            allow_pickle=True
        )

        self.features = model["features"]
        self.mean = model["mean"]
        self.std = model["std"]
        self.weights = model["weights"]
        self.bias = float(model["bias"])

    def predict(self, X):

        X = np.asarray(
            X,
            dtype=np.float32
        )

        X = X[:, self.features]

        X = (
            X - self.mean
        ) / self.std

        prediction = (
            X @ self.weights
            + self.bias
        )

        return np.clip(
            prediction,
            -1.0,
            1.0
        )


def generate_sample(brain, position):

    valid = np.where(
        np.isfinite(brain.azimuth)
    )[0]

    distance = np.abs(
        brain.azimuth[valid]
        - position
    )

    selected = valid[
        np.argsort(distance)[:500]
    ]

    visual = np.zeros(
        len(brain.azimuth),
        dtype=np.float32
    )

    visual[selected] = 1.0

    brain.reset()

    windows = []

    for _ in range(5):

        window = np.zeros(
            brain.n,
            dtype=np.float32
        )

        for _ in range(10):

            fired = brain.step(
                eye_drive=visual
            )

            if len(fired) > 0:

                window[fired] += 1

        windows.append(window)

    return np.concatenate(windows)


def main():

    print()
    print("=" * 60)
    print("FLY-001 CONTINUOUS DECODER HELD-OUT TEST")
    print("=" * 60)

    print()
    print(
        "Loading trained continuous decoder..."
    )

    decoder = ContinuousDecoder()

    print(
        f"Testing {len(TEST_POSITIONS)} "
        f"unseen positions"
    )

    print(
        f"Trials per position: "
        f"{TRIALS_PER_POSITION}"
    )

    brain = FlyBrain(
        device="auto"
    )

    targets = []

    predictions = []

    for position_index, position in enumerate(
        TEST_POSITIONS
    ):

        print()
        print(
            f"Position "
            f"{position_index + 1}/"
            f"{len(TEST_POSITIONS)}: "
            f"{position:+.2f}"
        )

        position_predictions = []

        for trial in range(
            TRIALS_PER_POSITION
        ):

            sample = generate_sample(
                brain,
                position
            )

            prediction = decoder.predict(
                sample.reshape(1, -1)
            )[0]

            targets.append(position)

            predictions.append(
                prediction
            )

            position_predictions.append(
                prediction
            )

            print(
                f"  Trial {trial + 1}: "
                f"{prediction:+.3f}"
            )

        average = np.mean(
            position_predictions
        )

        print(
            f"  Average: {average:+.3f}"
        )

    targets = np.asarray(
        targets,
        dtype=np.float32
    )

    predictions = np.asarray(
        predictions,
        dtype=np.float32
    )

    mae = np.mean(
        np.abs(
            predictions - targets
        )
    )

    rmse = np.sqrt(
        np.mean(
            (predictions - targets) ** 2
        )
    )

    correlation = np.corrcoef(
        predictions,
        targets
    )[0, 1]

    print()
    print("=" * 60)
    print("HELD-OUT RESULTS")
    print("=" * 60)

    print(
        f"Mean absolute error : "
        f"{mae:.4f}"
    )

    print(
        f"RMSE                : "
        f"{rmse:.4f}"
    )

    print(
        f"Correlation         : "
        f"{correlation:.4f}"
    )

    print()
    print(
        "POSITION COMPARISON"
    )

    print("-" * 60)

    for position in TEST_POSITIONS:

        mask = np.isclose(
            targets,
            position
        )

        average_prediction = np.mean(
            predictions[mask]
        )

        error = abs(
            average_prediction
            - position
        )

        print(
            f"Target {position:+.2f} "
            f"→ Prediction "
            f"{average_prediction:+.3f} "
            f"(error {error:.3f})"
        )

    print()
    print("=" * 60)


if __name__ == "__main__":

    main()