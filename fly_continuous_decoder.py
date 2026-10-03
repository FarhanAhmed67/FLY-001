import numpy as np
from flybrain import FlyBrain


OUTPUT_FILE = "fly_continuous_decoder.npz"

POSITIONS = np.linspace(-1.0, 1.0, 21)

TRIALS_PER_POSITION = 3

TOP_FEATURES = 1000


class ContinuousDecoder:

    def __init__(self, top_features=1000):

        self.top_features = top_features
        self.features = None
        self.mean = None
        self.std = None
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        X = np.asarray(
            X,
            dtype=np.float32
        )

        y = np.asarray(
            y,
            dtype=np.float32
        )

        print()
        print("Selecting neural features...")

        overall_mean = X.mean(axis=0)

        scores = np.mean(
            (X - overall_mean) ** 2,
            axis=0
        )

        count = min(
            self.top_features,
            X.shape[1]
        )

        self.features = np.argpartition(
            scores,
            -count
        )[-count:]

        X = X[:, self.features]

        self.mean = X.mean(axis=0)

        self.std = X.std(axis=0)

        self.std[self.std < 1e-6] = 1.0

        X = (
            X - self.mean
        ) / self.std

        print(
            f"Training linear continuous decoder "
            f"with {X.shape[1]} features..."
        )

        X_augmented = np.column_stack(
            [
                X,
                np.ones(len(X))
            ]
        )

        ridge = 1.0

        identity = np.eye(
            X_augmented.shape[1],
            dtype=np.float32
        )

        identity[-1, -1] = 0.0

        left = (
            X_augmented.T
            @ X_augmented
            + ridge * identity
        )

        right = (
            X_augmented.T
            @ y
        )

        solution = np.linalg.solve(
            left,
            right
        )

        self.weights = solution[:-1]

        self.bias = float(
            solution[-1]
        )

        return self

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
    print("FLY-001 CONTINUOUS NEURAL DECODER TRAINING")
    print("=" * 60)

    print()
    print(
        f"Positions: {len(POSITIONS)}"
    )

    print(
        f"Trials per position: "
        f"{TRIALS_PER_POSITION}"
    )

    total_samples = (
        len(POSITIONS)
        * TRIALS_PER_POSITION
    )

    print(
        f"Total training samples: "
        f"{total_samples}"
    )

    print()
    print(
        "Loading 166,700-neuron FlyBrain..."
    )

    brain = FlyBrain(
        device="auto"
    )

    X = []

    y = []

    for position_index, position in enumerate(
        POSITIONS
    ):

        print()
        print(
            f"Position {position_index + 1}/"
            f"{len(POSITIONS)}: "
            f"{position:+.2f}"
        )

        for trial in range(
            TRIALS_PER_POSITION
        ):

            sample = generate_sample(
                brain,
                position
            )

            X.append(sample)

            y.append(position)

            print(
                f"  Trial {trial + 1}/"
                f"{TRIALS_PER_POSITION} complete"
            )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.float32
    )

    print()
    print(
        f"Dataset shape: {X.shape}"
    )

    decoder = ContinuousDecoder(
        top_features=TOP_FEATURES
    )

    decoder.fit(
        X,
        y
    )

    predictions = decoder.predict(
        X
    )

    mae = np.mean(
        np.abs(
            predictions - y
        )
    )

    correlation = np.corrcoef(
        predictions,
        y
    )[0, 1]

    print()
    print("=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)

    print(
        f"Mean absolute error : "
        f"{mae:.4f}"
    )

    print(
        f"Correlation         : "
        f"{correlation:.4f}"
    )

    print()
    print(
        "Sample predictions:"
    )

    for i in range(
        min(10, len(y))
    ):

        print(
            f"Target {y[i]:+6.2f} "
            f"→ Predicted "
            f"{predictions[i]:+6.2f}"
        )

    np.savez(
        OUTPUT_FILE,
        features=decoder.features,
        mean=decoder.mean,
        std=decoder.std,
        weights=decoder.weights,
        bias=np.array(
            decoder.bias,
            dtype=np.float32
        )
    )

    print()
    print("=" * 60)

    print(
        f"Saved decoder: "
        f"{OUTPUT_FILE}"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()