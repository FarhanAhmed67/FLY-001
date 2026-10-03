import numpy as np

from flybrain import FlyBrain
from fly_decoder import FlyDecoder


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

N_TRIALS_PER_CLASS = 20
N_STEPS = 50
WINDOW_SIZE = 10
N_WINDOWS = 5

POSITIONS = {
    "STATIC": [1] * 50,
    "PULSE": (
        [1] * 10 +
        [0] * 10 +
        [1] * 10 +
        [0] * 20
    ),
    "DOUBLE_PULSE": (
        [1] * 6 +
        [0] * 9 +
        [1] * 6 +
        [0] * 29
    ),
    "SUSTAINED": (
        [1] * 25 +
        [0] * 25
    ),
}

CLASSES = list(POSITIONS.keys())


# ---------------------------------------------------------
# Load brain
# ---------------------------------------------------------

print("Initializing FlyBrain...")

brain = FlyBrain(device="auto")

print(f"Neurons: {len(brain.cell_type)}")
print(f"Visual receptors: {len(brain.visual)}")


# ---------------------------------------------------------
# Build exactly 500-receptor center stimulus
# ---------------------------------------------------------

azimuth = np.asarray(brain.azimuth)

direction = 0.0
distance = np.abs(azimuth - direction)

valid = np.where(np.isfinite(distance))[0]

selected = valid[
    np.argsort(distance[valid])[:500]
]

base_stimulus = np.zeros(
    len(azimuth),
    dtype=np.float32
)

base_stimulus[selected] = 1.0


# ---------------------------------------------------------
# Collect dataset
# ---------------------------------------------------------

X = []
y = []

print("\nCollecting temporal dataset...")

for class_index, class_name in enumerate(CLASSES):

    pattern = POSITIONS[class_name]

    print(
        f"  {class_name}: "
        f"{N_TRIALS_PER_CLASS} trials"
    )

    for trial in range(N_TRIALS_PER_CLASS):

        brain.reset()

        # 5 windows × 10 steps × 166700 neurons
        window_features = []

        for window in range(N_WINDOWS):

            window_activity = np.zeros(
                len(brain.cell_type),
                dtype=np.float32
            )

            start = window * WINDOW_SIZE
            end = start + WINDOW_SIZE

            for step in range(start, end):

                if pattern[step] == 1:
                    stimulus = base_stimulus
                else:
                    stimulus = np.zeros(
                        len(azimuth),
                        dtype=np.float32
                    )

                fired = brain.step(
                    eye_drive=stimulus
                )

                window_activity[fired] += 1

            window_features.append(
                window_activity
            )

        # Preserve temporal structure:
        # [window1 neurons,
        #  window2 neurons,
        #  ...
        #  window5 neurons]
        features = np.concatenate(
            window_features
        )

        X.append(features)
        y.append(class_index)


X = np.asarray(X, dtype=np.float32)
y = np.asarray(y)

print("\nDataset collected.")
print("X shape:", X.shape)
print("y shape:", y.shape)


# ---------------------------------------------------------
# Create stratified 5-fold split
# ---------------------------------------------------------

rng = np.random.default_rng(42)

fold_indices = [
    [] for _ in range(5)
]

for class_index in range(len(CLASSES)):

    indices = np.where(y == class_index)[0]

    rng.shuffle(indices)

    # 20 trials → 5 folds × 4 trials
    splits = np.array_split(indices, 5)

    for fold in range(5):
        fold_indices[fold].extend(
            splits[fold].tolist()
        )


# ---------------------------------------------------------
# 5-fold cross-validation
# ---------------------------------------------------------

accuracies = []

all_predictions = []
all_targets = []

print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION")
print("=" * 60)

for fold in range(5):

    test_indices = np.array(
        fold_indices[fold]
    )

    train_indices = np.concatenate(
        [
            np.array(fold_indices[i])
            for i in range(5)
            if i != fold
        ]
    )

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    print(f"\nFold {fold + 1}/5")
    print(
        f"Training samples: {len(train_indices)}"
    )
    print(
        f"Testing samples:  {len(test_indices)}"
    )

        # Feature selection happens here,
    # using TRAINING data only.
    decoder = FlyDecoder(
    classes=list(range(len(CLASSES))),
    top_features=1000
    )

    decoder.fit(
        X_train,
        y_train
    )

    predictions = decoder.predict(
        X_test
    )

    accuracy = np.mean(
        predictions == y_test
    )

    accuracies.append(accuracy)

    all_predictions.extend(
        predictions.tolist()
    )

    all_targets.extend(
        y_test.tolist()
    )

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    all_predictions.extend(
        predictions.tolist()
    )

    all_targets.extend(
        y_test.tolist()
    )



# ---------------------------------------------------------
# Overall results
# ---------------------------------------------------------

accuracies = np.asarray(
    accuracies
)

all_predictions = np.asarray(
    all_predictions
)

all_targets = np.asarray(
    all_targets
)

overall_accuracy = np.mean(
    all_predictions == all_targets
)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

for i, accuracy in enumerate(accuracies):

    print(
        f"Fold {i + 1}: "
        f"{accuracy * 100:.2f}%"
    )

print(
    f"\nMean accuracy: "
    f"{accuracies.mean() * 100:.2f}%"
)

print(
    f"Std deviation: "
    f"{accuracies.std() * 100:.2f}%"
)

print(
    f"Overall accuracy: "
    f"{overall_accuracy * 100:.2f}%"
)

print(
    "\nChance level: 25.00%"
)


# ---------------------------------------------------------
# Per-pattern results
# ---------------------------------------------------------

print("\nPer-pattern results:")

for class_index, class_name in enumerate(CLASSES):

    mask = all_targets == class_index

    accuracy = np.mean(
        all_predictions[mask]
        == all_targets[mask]
    )

    print(
        f"{class_name:15s}: "
        f"{accuracy * 100:.2f}%"
    )


print("\nTemporal cross-validation complete.")