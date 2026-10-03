import numpy as np
from flybrain import FlyBrain


TRAIN_POSITIONS = np.arange(-1.0, 1.01, 0.2)
TEST_POSITIONS = np.arange(-0.9, 0.91, 0.2)

TRIALS = 10
BRAIN_STEPS = 50
SIGMA = 0.08
TOP_FEATURES = 1000
RIDGE_ALPHA = 10.0


def soft_encode(azimuth, position, sigma):
    azimuth = np.asarray(azimuth, dtype=np.float32)

    distance = np.abs(azimuth - position)

    stimulus = np.exp(
        -(distance ** 2) / (2.0 * sigma ** 2)
    )

    stimulus[~np.isfinite(azimuth)] = 0.0

    maximum = stimulus.max()

    if maximum > 0:
        stimulus = stimulus / maximum

    return stimulus.astype(np.float32)


def collect_sample(brain, stimulus):
    brain.reset()

    spike_counts = np.zeros(
        len(brain.cell_type),
        dtype=np.float32
    )

    for _ in range(BRAIN_STEPS):
        fired = brain.step(
            eye_drive=stimulus
        )

        if len(fired) > 0:
            spike_counts += np.bincount(
                fired,
                minlength=len(brain.cell_type)
            )

    return spike_counts


def collect_dataset(brain, positions, trials):
    X = []
    y = []

    for position in positions:

        print(
            f"Collecting position {position:+.2f}"
        )

        stimulus = soft_encode(
            brain.azimuth,
            position,
            SIGMA
        )

        for trial in range(trials):

            sample = collect_sample(
                brain,
                stimulus
            )

            X.append(sample)
            y.append(position)

            print(
                f"  trial {trial + 1}/{trials} "
                f"| spikes {int(sample.sum())}"
            )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32)
    )


def select_features(X, y, count):
    overall_mean = X.mean(axis=0)

    scores = np.zeros(
        X.shape[1],
        dtype=np.float32
    )

    for position in np.unique(y):

        mask = np.isclose(
            y,
            position,
            atol=1e-5
        )

        if not np.any(mask):
            continue

        class_mean = X[mask].mean(axis=0)

        scores += (
            np.sum(mask)
            * (class_mean - overall_mean) ** 2
        )

    count = min(
        count,
        X.shape[1]
    )

    features = np.argpartition(
        scores,
        -count
    )[-count:]

    return features


def standardize_train_test(
    X_train,
    X_test,
    features
):
    X_train = X_train[:, features]
    X_test = X_test[:, features]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    std[std < 1e-6] = 1.0

    X_train = (
        X_train - mean
    ) / std

    X_test = (
        X_test - mean
    ) / std

    return (
        X_train,
        X_test,
        mean,
        std
    )


def fit_ridge(X, y, alpha):
    n_features = X.shape[1]

    A = (
        X.T @ X
        + alpha * np.eye(
            n_features,
            dtype=np.float32
        )
    )

    b = X.T @ y

    weights = np.linalg.solve(
        A,
        b
    )

    return weights


def evaluate(
    name,
    X,
    y,
    weights
):
    predictions = X @ weights

    mae = np.mean(
        np.abs(predictions - y)
    )

    rmse = np.sqrt(
        np.mean(
            (predictions - y) ** 2
        )
    )

    if (
        np.std(predictions) > 0
        and np.std(y) > 0
    ):
        correlation = np.corrcoef(
            predictions,
            y
        )[0, 1]
    else:
        correlation = 0.0

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)
    print()

    print(
        f"MAE         : {mae:.4f}"
    )

    print(
        f"RMSE        : {rmse:.4f}"
    )

    print(
        f"Correlation : {correlation:.4f}"
    )

    return predictions


def main():

    print()
    print("=" * 60)
    print("FLY-001 SOFT CONTINUOUS BRAIN DECODER")
    print("=" * 60)

    print()
    print("Training positions :", len(TRAIN_POSITIONS))
    print("Testing positions  :", len(TEST_POSITIONS))
    print("Trials per position:", TRIALS)
    print("Brain steps/sample :", BRAIN_STEPS)
    print("Sigma              :", SIGMA)
    print("Top features       :", TOP_FEATURES)
    print("Ridge alpha        :", RIDGE_ALPHA)

    print()
    print("Loading FlyBrain...")

    brain = FlyBrain(
        device="auto"
    )

    print(
        "Neurons            :",
        len(brain.cell_type)
    )

    print(
        "Visual receptors   :",
        len(brain.azimuth)
    )

    print()
    print("=" * 60)
    print("COLLECTING TRAINING DATA")
    print("=" * 60)

    X_train, y_train = collect_dataset(
        brain,
        TRAIN_POSITIONS,
        TRIALS
    )

    print()
    print(
        "Training samples:",
        len(X_train)
    )

    print()
    print("=" * 60)
    print("COLLECTING HELD-OUT DATA")
    print("=" * 60)

    X_test, y_test = collect_dataset(
        brain,
        TEST_POSITIONS,
        TRIALS
    )

    print()
    print(
        "Testing samples:",
        len(X_test)
    )

    print()
    print("=" * 60)
    print("SELECTING FEATURES")
    print("=" * 60)

    features = select_features(
        X_train,
        y_train,
        TOP_FEATURES
    )

    print(
        "Selected features:",
        len(features)
    )

    (
        X_train_scaled,
        X_test_scaled,
        mean,
        std
    ) = standardize_train_test(
        X_train,
        X_test,
        features
    )

    print()
    print("=" * 60)
    print("TRAINING RIDGE DECODER")
    print("=" * 60)

    weights = fit_ridge(
        X_train_scaled,
        y_train,
        RIDGE_ALPHA
    )

    train_predictions = evaluate(
        "TRAINING PERFORMANCE",
        X_train_scaled,
        y_train,
        weights
    )

    test_predictions = evaluate(
        "HELD-OUT PERFORMANCE",
        X_test_scaled,
        y_test,
        weights
    )

    print()
    print("=" * 60)
    print("HELD-OUT POSITION RESULTS")
    print("=" * 60)
    print()

    for position in TEST_POSITIONS:

        mask = np.isclose(
            y_test,
            position,
            atol=1e-5
        )

        actual = y_test[mask]
        predicted = test_predictions[mask]

        if len(actual) == 0:
            continue

        mean_prediction = predicted.mean()
        std_prediction = predicted.std()
        error = abs(
            mean_prediction
            - actual.mean()
        )

        print(
            f"{position:+.2f} -> "
            f"mean {mean_prediction:+.3f} "
            f"| std {std_prediction:.3f} "
            f"| error {error:.3f}"
        )

    np.savez(
        "fly_soft_continuous_decoder.npz",
        features=features,
        mean=mean,
        std=std,
        weights=weights,
        train_positions=TRAIN_POSITIONS,
        test_positions=TEST_POSITIONS,
        sigma=SIGMA
    )

    print()
    print("=" * 60)
    print("MODEL SAVED")
    print("=" * 60)
    print()

    print(
        "fly_soft_continuous_decoder.npz"
    )

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()