import numpy as np
from flybrain import FlyBrain


TRAIN_POSITIONS = np.arange(-1.0, 1.01, 0.2)
TEST_POSITIONS = np.arange(-0.9, 1.0, 0.2)

TRIALS = 10
STEPS = 50

POPULATION_SIZES = [20, 50, 100]


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
            f"  Completed {TRIALS} trials"
        )

    return (
        np.asarray(data),
        np.asarray(labels)
    )


def select_tuned_neurons(X, y, count):

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
        count,
        X.shape[1]
    )

    indices = np.argpartition(
        scores,
        -count
    )[-count:]

    # Sort by score so the most tuned neurons
    # appear first.

    indices = indices[
        np.argsort(
            scores[indices]
        )[::-1]
    ]

    return indices


def fit_ridge(X, y, alpha=10.0):

    mean = X.mean(
        axis=0
    )

    std = X.std(
        axis=0
    )

    std[std < 1e-6] = 1.0

    X_norm = (
        X - mean
    ) / std

    X_aug = np.column_stack(
        [
            np.ones(len(X_norm)),
            X_norm
        ]
    )

    identity = np.eye(
        X_aug.shape[1]
    )

    identity[0, 0] = 0.0

    weights = np.linalg.solve(
        X_aug.T @ X_aug
        + alpha * identity,
        X_aug.T @ y
    )

    return (
        mean,
        std,
        weights
    )


def predict(
    X,
    mean,
    std,
    weights
):

    X_norm = (
        X - mean
    ) / std

    X_aug = np.column_stack(
        [
            np.ones(len(X_norm)),
            X_norm
        ]
    )

    return X_aug @ weights


def evaluate(
    name,
    actual,
    predicted
):

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
    print(name)

    print(
        f"  MAE         : {mae:.4f}"
    )

    print(
        f"  RMSE        : {rmse:.4f}"
    )

    print(
        f"  Correlation : {correlation:.4f}"
    )

    return (
        mae,
        rmse,
        correlation
    )


def main():

    print()
    print("=" * 70)
    print("FLY-001 POPULATION POSITION DECODER")
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
    # TEST DATA
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("COLLECTING HELD-OUT TEST DATA")
    print("=" * 70)

    X_test_full, y_test = collect_data(
        brain,
        TEST_POSITIONS
    )

    print()
    print(
        f"Test samples: "
        f"{len(X_test_full)}"
    )

    # --------------------------------------------------
    # TEST DIFFERENT POPULATION SIZES
    # --------------------------------------------------

    results = {}

    for population_size in POPULATION_SIZES:

        print()
        print("=" * 70)
        print(
            f"POPULATION SIZE: "
            f"{population_size}"
        )
        print("=" * 70)

        tuned_indices = select_tuned_neurons(
            X_train_full,
            y_train,
            population_size
        )

        X_train = X_train_full[
            :,
            tuned_indices
        ]

        X_test = X_test_full[
            :,
            tuned_indices
        ]

        print()
        print(
            f"Using {len(tuned_indices)} "
            "training-selected neurons."
        )

        mean, std, weights = fit_ridge(
            X_train,
            y_train
        )

        train_pred = predict(
            X_train,
            mean,
            std,
            weights
        )

        test_pred = predict(
            X_test,
            mean,
            std,
            weights
        )

        train_results = evaluate(
            "TRAINING",
            y_train,
            train_pred
        )

        test_results = evaluate(
            "HELD-OUT",
            y_test,
            test_pred
        )

        results[
            population_size
        ] = {
            "indices": tuned_indices,
            "train": train_results,
            "test": test_results
        }

        # --------------------------------------------------
        # POSITION MEANS
        # --------------------------------------------------

        print()
        print(
            "Held-out position means:"
        )

        print()

        print(
            f"{'Actual':<10}"
            f"{'Predicted':<15}"
            f"{'Error':<10}"
        )

        print("-" * 35)

        for position in TEST_POSITIONS:

            mask = (
                y_test == position
            )

            mean_prediction = (
                test_pred[mask].mean()
            )

            error = abs(
                position
                - mean_prediction
            )

            print(
                f"{position:+.2f}"
                f"{'':<6}"
                f"{mean_prediction:+.3f}"
                f"{'':<8}"
                f"{error:.3f}"
            )

    # --------------------------------------------------
    # FINAL COMPARISON
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("POPULATION SIZE COMPARISON")
    print("=" * 70)

    print()

    print(
        f"{'Neurons':<10}"
        f"{'Train MAE':<12}"
        f"{'Test MAE':<12}"
        f"{'Test RMSE':<12}"
        f"{'Test Corr':<12}"
    )

    print("-" * 60)

    for population_size in POPULATION_SIZES:

        train_results = results[
            population_size
        ]["train"]

        test_results = results[
            population_size
        ]["test"]

        print(
            f"{population_size:<10}"
            f"{train_results[0]:<12.4f}"
            f"{test_results[0]:<12.4f}"
            f"{test_results[1]:<12.4f}"
            f"{test_results[2]:<12.4f}"
        )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    save_data = {
        "train_positions":
            TRAIN_POSITIONS,

        "test_positions":
            TEST_POSITIONS,

        "y_train":
            y_train,

        "y_test":
            y_test
    }

    for population_size in POPULATION_SIZES:

        result = results[
            population_size
        ]

        save_data[
            f"indices_{population_size}"
        ] = result["indices"]

        save_data[
            f"train_metrics_{population_size}"
        ] = np.asarray(
            result["train"]
        )

        save_data[
            f"test_metrics_{population_size}"
        ] = np.asarray(
            result["test"]
        )

    np.savez(
        "fly_population_position_decoder.npz",
        **save_data
    )

    print()
    print("=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        "fly_population_position_decoder.npz"
    )

    print()
    print("=" * 70)
    print("POPULATION POSITION DECODER COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()