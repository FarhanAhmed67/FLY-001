import numpy as np

from flybrain import FlyBrain
from fly_decoder import FlyDecoder


# =========================================================
# CONFIGURATION
# =========================================================

POSITIONS = {
    "LEFT": -0.75,
    "CENTER_LEFT": -0.375,
    "CENTER": 0.0,
    "CENTER_RIGHT": 0.375,
    "RIGHT": 0.75,
}

INTENSITIES = {
    "DIM": 0.25,
    "MEDIUM": 0.50,
    "BRIGHT": 1.00,
}

TEMPORAL_PATTERNS = {
    "STATIC": (
        [1] * 50
    ),

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

TRIALS_PER_COMBINATION = 5

N_STEPS = 50
WINDOW_SIZE = 10
N_WINDOWS = 5
N_RECEPTORS = 500


# =========================================================
# BUILD COMBINATION LABELS
# =========================================================

combinations = []

for position in POSITIONS:
    for intensity in INTENSITIES:
        for temporal in TEMPORAL_PATTERNS:

            combinations.append(
                (
                    position,
                    intensity,
                    temporal
                )
            )

print("=" * 70)
print("FLY-001 3D COMPOSITIONAL TEST")
print("=" * 70)

print(
    f"Positions: {len(POSITIONS)}"
)

print(
    f"Intensities: {len(INTENSITIES)}"
)

print(
    f"Temporal patterns: "
    f"{len(TEMPORAL_PATTERNS)}"
)

print(
    f"Combinations: {len(combinations)}"
)

print(
    f"Trials per combination: "
    f"{TRIALS_PER_COMBINATION}"
)

print(
    f"Total trials: "
    f"{len(combinations) * TRIALS_PER_COMBINATION}"
)


# =========================================================
# LOAD BRAIN
# =========================================================

print("\nInitializing FlyBrain...")

brain = FlyBrain(device="auto")

print(
    f"Neurons: {len(brain.cell_type)}"
)

print(
    f"Visual receptors: {len(brain.visual)}"
)


# =========================================================
# PRECOMPUTE 500 RECEPTOR MASKS
# =========================================================

azimuth = np.asarray(
    brain.azimuth
)

position_masks = {}

valid = np.where(
    np.isfinite(azimuth)
)[0]

for position, direction in POSITIONS.items():

    distance = np.abs(
        azimuth - direction
    )

    selected = valid[
        np.argsort(
            distance[valid]
        )[:N_RECEPTORS]
    ]

    mask = np.zeros(
        len(azimuth),
        dtype=np.float32
    )

    mask[selected] = 1.0

    position_masks[position] = mask


# =========================================================
# COLLECT DATA
# =========================================================

X = []
y = []

print("\n" + "=" * 70)
print("COLLECTING DATA")
print("=" * 70)

for class_index, combination in enumerate(
    combinations
):

    position, intensity, temporal = combination

    pattern = TEMPORAL_PATTERNS[
        temporal
    ]

    base_stimulus = (
        position_masks[position]
        * INTENSITIES[intensity]
    )

    print(
        f"\n[{class_index + 1:02d}/"
        f"{len(combinations)}] "
        f"{position} + "
        f"{intensity} + "
        f"{temporal}"
    )

    for trial in range(
        TRIALS_PER_COMBINATION
    ):

        brain.reset()

        window_features = []

        for window in range(
            N_WINDOWS
        ):

            window_activity = np.zeros(
                len(brain.cell_type),
                dtype=np.float32
            )

            start = (
                window * WINDOW_SIZE
            )

            end = (
                start + WINDOW_SIZE
            )

            for step in range(
                start,
                end
            ):

                if pattern[step] == 1:

                    stimulus = (
                        base_stimulus
                    )

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

        features = np.concatenate(
            window_features
        )

        X.append(features)
        y.append(class_index)


X = np.asarray(
    X,
    dtype=np.float32
)

y = np.asarray(y)


print("\n" + "=" * 70)
print("DATA COLLECTION COMPLETE")
print("=" * 70)

print(
    "Feature matrix:",
    X.shape
)

print(
    "Labels:",
    y.shape
)


# =========================================================
# 5-FOLD STRATIFIED CROSS-VALIDATION
# =========================================================

rng = np.random.default_rng(42)

fold_indices = [
    [] for _ in range(5)
]

for class_index in range(
    len(combinations)
):

    indices = np.where(
        y == class_index
    )[0]

    rng.shuffle(indices)

    splits = np.array_split(
        indices,
        5
    )

    for fold in range(5):

        fold_indices[fold].extend(
            splits[fold].tolist()
        )


# =========================================================
# CROSS-VALIDATION
# =========================================================

accuracies = []

all_predictions = []
all_targets = []

print("\n" + "=" * 70)
print("5-FOLD 3D CROSS-VALIDATION")
print("=" * 70)

for fold in range(5):

    test_indices = np.array(
        fold_indices[fold]
    )

    train_indices = np.concatenate(
        [
            np.array(
                fold_indices[i]
            )
            for i in range(5)
            if i != fold
        ]
    )

    X_train = X[
        train_indices
    ]

    y_train = y[
        train_indices
    ]

    X_test = X[
        test_indices
    ]

    y_test = y[
        test_indices
    ]

    print(
        f"\nFold {fold + 1}/5"
    )

    print(
        f"Training samples: "
        f"{len(train_indices)}"
    )

    print(
        f"Testing samples: "
        f"{len(test_indices)}"
    )

    decoder = FlyDecoder(
        classes=list(
            range(len(combinations))
        ),
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

    accuracies.append(
        accuracy
    )

    all_predictions.extend(
        predictions.tolist()
    )

    all_targets.extend(
        y_test.tolist()
    )

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )


# =========================================================
# FINAL RESULTS
# =========================================================

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

print("\n" + "=" * 70)
print("FINAL 3D COMPOSITIONAL RESULTS")
print("=" * 70)

for i, accuracy in enumerate(
    accuracies
):

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
    "\nChance level: "
    f"{100 / len(combinations):.2f}%"
)


# =========================================================
# FACTOR-LEVEL ANALYSIS
# =========================================================

print("\n" + "=" * 70)
print("FACTOR-LEVEL RESULTS")
print("=" * 70)


predicted_combinations = [
    combinations[int(index)]
    for index in all_predictions
]

target_combinations = [
    combinations[int(index)]
    for index in all_targets
]


position_correct = 0
intensity_correct = 0
temporal_correct = 0

for predicted, target in zip(
    predicted_combinations,
    target_combinations
):

    if predicted[0] == target[0]:
        position_correct += 1

    if predicted[1] == target[1]:
        intensity_correct += 1

    if predicted[2] == target[2]:
        temporal_correct += 1


total = len(
    target_combinations
)

print(
    f"Position accuracy: "
    f"{position_correct / total * 100:.2f}%"
)

print(
    f"Intensity accuracy: "
    f"{intensity_correct / total * 100:.2f}%"
)

print(
    f"Temporal accuracy: "
    f"{temporal_correct / total * 100:.2f}%"
)


print("\n3D compositional test complete.")