import numpy as np
from flybrain import FlyBrain


TRAIN_POSITIONS = np.linspace(-1.0, 1.0, 11)

TEST_POSITIONS = np.arange(
    -0.9,
    1.0,
    0.2
)

TRIALS = 10

TOP_FEATURES = 1000


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


def collect(brain, positions):

    X = []

    y = []

    for position in positions:

        print()
        print(
            f"Position {position:+.2f}"
        )

        for trial in range(TRIALS):

            sample = generate_sample(
                brain,
                position
            )

            X.append(sample)

            y.append(position)

            print(
                f"  Trial "
                f"{trial + 1}/{TRIALS}"
            )

    return (
        np.asarray(
            X,
            dtype=np.float32
        ),
        np.asarray(
            y,
            dtype=np.float32
        )
    )


def select_features(X, y):

    print()
    print(
        "Selecting position-sensitive neurons..."
    )

    overall_mean = X.mean(
        axis=0
    )

    scores = np.zeros(
        X.shape[1],
        dtype=np.float32
    )

    for position in np.unique(y):

        mask = y == position

        class_data = X[mask]

        class_mean = (
            class_data.mean(axis=0)
        )

        difference = (
            class_mean
            - overall_mean
        )

        scores += (
            len(class_data)
            * difference ** 2
        )

    count = min(
        TOP_FEATURES,
        X.shape[1]
    )

    features = np.argpartition(
        scores,
        -count
    )[-count:]

    return features


def train_decoder(X, y, features):

    X = X[:, features]

    mean = X.mean(
        axis=0
    )

    std = X.std(
        axis=0
    )

    std[std < 1e-6] = 1.0

    X = (
        X - mean
    ) / std

    X_aug = np.column_stack(
        [
            X,
            np.ones(len(X))
        ]
    )

    ridge = 10.0

    identity = np.eye(
        X_aug.shape[1],
        dtype=np.float32
    )

    identity[-1, -1] = 0.0

    solution = np.linalg.solve(
        X_aug.T @ X_aug
        + ridge * identity,
        X_aug.T @ y
    )

    weights = solution[:-1]

    bias = solution[-1]

    return (
        mean,
        std,
        weights,
        bias
    )


def predict(
    X,
    features,
    mean,
    std,
    weights,
    bias
):

    X = X[:, features]

    X = (
        X - mean
    ) / std

    predictions = (
        X @ weights
        + bias
    )

    return np.clip(
        predictions,
        -1.0,
        1.0
    )


def main():

    print()
    print("=" * 60)
    print(
        "FLY-001 SPATIAL GENERALIZATION TEST"
    )
    print("=" * 60)

    print()
    print(
        "Training positions:"
    )

    print(
        " ".join(
            f"{x:+.1f}"
            for x in TRAIN_POSITIONS
        )
    )

    print()
    print(
        "Held-out positions:"
    )

    print(
        " ".join(
            f"{x:+.1f}"
            for x in TEST_POSITIONS
        )
    )

    print()
    print(
        f"Trials per position: {TRIALS}"
    )

    brain = FlyBrain(
        device="auto"
    )

    print()
    print(
        "Collecting TRAINING data..."
    )

    X_train, y_train = collect(
        brain,
        TRAIN_POSITIONS
    )

    print()
    print(
        "Collecting HELD-OUT data..."
    )

    X_test, y_test = collect(
        brain,
        TEST_POSITIONS
    )

    print()
    print(
        f"Training shape: {X_train.shape}"
    )

    print(
        f"Test shape    : {X_test.shape}"
    )

    features = select_features(
        X_train,
        y_train
    )

    (
        mean,
        std,
        weights,
        bias
    ) = train_decoder(
        X_train,
        y_train,
        features
    )

    train_predictions = predict(
        X_train,
        features,
        mean,
        std,
        weights,
        bias
    )

    test_predictions = predict(
        X_test,
        features,
        mean,
        std,
        weights,
        bias
    )

    train_mae = np.mean(
        np.abs(
            train_predictions
            - y_train
        )
    )

    test_mae = np.mean(
        np.abs(
            test_predictions
            - y_test
        )
    )

    train_corr = np.corrcoef(
        train_predictions,
        y_train
    )[0, 1]

    test_corr = np.corrcoef(
        test_predictions,
        y_test
    )[0, 1]

    print()
    print("=" * 60)
    print(
        "GENERALIZATION RESULTS"
    )
    print("=" * 60)

    print()
    print(
        f"Training MAE       : "
        f"{train_mae:.4f}"
    )

    print(
        f"Training correlation: "
        f"{train_corr:.4f}"
    )

    print()

    print(
        f"Held-out MAE       : "
        f"{test_mae:.4f}"
    )

    print(
        f"Held-out correlation: "
        f"{test_corr:.4f}"
    )

    print()
    print(
        "HELD-OUT PREDICTIONS"
    )

    print(
        "-" * 60
    )

    for target in TEST_POSITIONS:

        mask = np.isclose(
            y_test,
            target
        )

        average = np.mean(
            test_predictions[mask]
        )

        error = abs(
            average - target
        )

        print(
            f"{target:+.2f} "
            f"→ {average:+.3f} "
            f"(error {error:.3f})"
        )

    print()
    print("=" * 60)


if __name__ == "__main__":

    main()