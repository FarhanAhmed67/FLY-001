import numpy as np
from flybrain import FlyBrain


TRAIN_POSITIONS = np.arange(-1.0, 1.01, 0.2)
TEST_POSITIONS = np.arange(-0.9, 1.0, 0.2)

TRIALS = 10
STEPS = 50

TOP_N = 20


def create_visual(brain, position):

    valid = np.where(
        np.isfinite(brain.azimuth)
    )[0]

    distance = np.abs(
        brain.azimuth[valid] - position
    )

    selected = valid[
        np.argsort(distance)[:500]
    ]

    visual = np.zeros(
        len(brain.azimuth),
        dtype=np.float32
    )

    visual[selected] = 1.0

    return visual


def measure_activity(brain, visual):

    brain.reset()

    counts = np.zeros(
        len(brain.positions),
        dtype=np.float32
    )

    for _ in range(STEPS):

        fired = brain.step(
            eye_drive=visual
        )

        if len(fired) > 0:

            counts += np.bincount(
                fired,
                minlength=len(brain.positions)
            )

    return counts


def collect_data(brain, positions):

    data = []
    labels = []

    for p_index, position in enumerate(positions):

        print()
        print(
            f"Position {p_index + 1}/"
            f"{len(positions)}: "
            f"{position:+.2f}"
        )

        visual = create_visual(
            brain,
            position
        )

        for trial in range(TRIALS):

            counts = measure_activity(
                brain,
                visual
            )

            data.append(counts)
            labels.append(position)

        print(
            "  Completed "
            f"{TRIALS} trials"
        )

    return (
        np.asarray(data),
        np.asarray(labels)
    )


def select_tuned_neurons(X, y):

    overall_mean = X.mean(
        axis=0
    )

    scores = np.zeros(
        X.shape[1],
        dtype=np.float32
    )

    for position in TRAIN_POSITIONS:

        mask = y == position

        if not np.any(mask):
            continue

        class_data = X[mask]

        class_mean = class_data.mean(
            axis=0
        )

        scores += len(class_data) * (
            class_mean - overall_mean
        ) ** 2

    count = min(
        TOP_N,
        X.shape[1]
    )

    indices = np.argpartition(
        scores,
        -count
    )[-count:]

    return indices


def fit_ridge(X, y, alpha=10.0):

    X_mean = X.mean(
        axis=0
    )

    X_std = X.std(
        axis=0
    )

    X_std[X_std < 1e-6] = 1.0

    X_norm = (
        X - X_mean
    ) / X_std

    X_aug = np.column_stack(
        [
            np.ones(len(X_norm)),
            X_norm
        ]
    )

    I = np.eye(
        X_aug.shape[1]
    )

    I[0, 0] = 0.0

    weights = np.linalg.solve(
        X_aug.T @ X_aug
        + alpha * I,
        X_aug.T @ y
    )

    return (
        X_mean,
        X_std,
        weights
    )


def predict(
    X,
    X_mean,
    X_std,
    weights
):

    X_norm = (
        X - X_mean
    ) / X_std

    X_aug = np.column_stack(
        [
            np.ones(len(X_norm)),
            X_norm
        ]
    )

    return X_aug @ weights


def evaluate(name, actual, predicted):

    mae = np.mean(
        np.abs(
            actual - predicted
        )
    )

    rmse = np.sqrt(
        np.mean(
            (actual - predicted) ** 2
        )
    )

    if (
        np.std(actual) > 0
        and np.std(predicted) > 0
    ):

        correlation = np.corrcoef(
            actual,
            predicted
        )[0, 1]

    else:

        correlation = 0.0

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"MAE         : {mae:.4f}"
    )

    print(
        f"RMSE        : {rmse:.4f}"
    )

    print(
        f"Correlation : {correlation:.4f}"
    )

    return mae, rmse, correlation


def main():

    print()
    print("=" * 70)
    print("FLY-001 TUNED NEURON POSITION DECODER")
    print("=" * 70)

    print()
    print("Training positions:")
    print(
        " ".join(
            f"{x:+.1f}"
            for x in TRAIN_POSITIONS
        )
    )

    print()
    print("Held-out positions:")
    print(
        " ".join(
            f"{x:+.1f}"
            for x in TEST_POSITIONS
        )
    )

    print()
    print(
        f"Trials per position : {TRIALS}"
    )

    print(
        f"Brain steps         : {STEPS}"
    )

    print(
        f"Selected neurons    : {TOP_N}"
    )

    brain = FlyBrain(
        device="auto"
    )

    print()
    print(
        f"Neurons: {len(brain.positions)}"
    )

    # --------------------------------------------------
    # TRAINING DATA
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("COLLECTING TRAINING DATA")
    print("=" * 70)

    X_train_full, y_train = collect_data(
        brain,
        TRAIN_POSITIONS
    )

    print()
    print(
        f"Training samples: "
        f"{len(X_train_full)}"
    )

    # --------------------------------------------------
    # SELECT TUNED NEURONS
    # --------------------------------------------------

    tuned_indices = select_tuned_neurons(
        X_train_full,
        y_train
    )

    X_train = X_train_full[
        :,
        tuned_indices
    ]

    print()
    print(
        f"Selected {len(tuned_indices)} "
        "training-only tuned neurons."
    )

    print()
    print("Selected neuron IDs:")

    print(
        " ".join(
            str(int(x))
            for x in tuned_indices
        )
    )

    # --------------------------------------------------
    # FIT DECODER
    # --------------------------------------------------

    X_mean, X_std, weights = fit_ridge(
        X_train,
        y_train,
        alpha=10.0
    )

    train_pred = predict(
        X_train,
        X_mean,
        X_std,
        weights
    )

    train_results = evaluate(
        "TRAINING PERFORMANCE",
        y_train,
        train_pred
    )

    # --------------------------------------------------
    # HELD-OUT TEST DATA
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("COLLECTING HELD-OUT TEST DATA")
    print("=" * 70)

    X_test_full, y_test = collect_data(
        brain,
        TEST_POSITIONS
    )

    X_test = X_test_full[
        :,
        tuned_indices
    ]

    test_pred = predict(
        X_test,
        X_mean,
        X_std,
        weights
    )

    test_results = evaluate(
        "HELD-OUT PERFORMANCE",
        y_test,
        test_pred
    )

    # --------------------------------------------------
    # POSITION-BY-POSITION RESULTS
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("HELD-OUT POSITION RESULTS")
    print("=" * 70)

    print()

    print(
        f"{'Actual':<10}"
        f"{'Predicted Mean':<18}"
        f"{'Absolute Error':<15}"
    )

    print("-" * 50)

    for position in TEST_POSITIONS:

        mask = y_test == position

        predictions = test_pred[
            mask
        ]

        mean_prediction = predictions.mean()

        error = abs(
            position
            - mean_prediction
        )

        print(
            f"{position:+.2f}"
            f"{'':<6}"
            f"{mean_prediction:+.3f}"
            f"{'':<13}"
            f"{error:.3f}"
        )

    # --------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------

    np.savez(
        "fly_tuned_position_decoder.npz",

        tuned_indices=tuned_indices,

        train_positions=TRAIN_POSITIONS,

        test_positions=TEST_POSITIONS,

        mean=X_mean,

        std=X_std,

        weights=weights,

        train_mae=train_results[0],

        train_rmse=train_results[1],

        train_correlation=train_results[2],

        test_mae=test_results[0],

        test_rmse=test_results[1],

        test_correlation=test_results[2]
    )

    print()
    print("=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        "fly_tuned_position_decoder.npz"
    )

    print()
    print("=" * 70)
    print("TUNED POSITION DECODER COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()