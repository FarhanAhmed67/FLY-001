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

TRIALS_PER_CONTEXT = 5

N_STEPS = 50
WINDOW_SIZE = 10
N_WINDOWS = 5
N_RECEPTORS = 500


# =========================================================
# LABELS
# =========================================================

position_names = list(
    POSITIONS.keys()
)

position_to_label = {
    name: index
    for index, name
    in enumerate(position_names)
}


print("=" * 70)
print("FLY-001 POSITION CONTEXT GENERALIZATION")
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
    f"Trials per context: "
    f"{TRIALS_PER_CONTEXT}"
)

total_trials = (
    len(POSITIONS)
    * len(INTENSITIES)
    * len(TEMPORAL_PATTERNS)
    * TRIALS_PER_CONTEXT
)

print(
    f"Total trials: {total_trials}"
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
# BUILD POSITION MASKS
# =========================================================

azimuth = np.asarray(
    brain.azimuth
)

valid = np.where(
    np.isfinite(azimuth)
)[0]

position_masks = {}

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

for position in POSITIONS:

    for intensity in INTENSITIES:

        for temporal in TEMPORAL_PATTERNS:

            pattern = TEMPORAL_PATTERNS[
                temporal
            ]

            stimulus = (
                position_masks[position]
                * INTENSITIES[intensity]
            )

            print(
                f"{position:12s} + "
                f"{intensity:6s} + "
                f"{temporal:13s}"
            )

            for trial in range(
                TRIALS_PER_CONTEXT
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
                        window
                        * WINDOW_SIZE
                    )

                    end = (
                        start
                        + WINDOW_SIZE
                    )

                    for step in range(
                        start,
                        end
                    ):

                        if pattern[step] == 1:

                            current_stimulus = (
                                stimulus
                            )

                        else:

                            current_stimulus = np.zeros(
                                len(azimuth),
                                dtype=np.float32
                            )

                        fired = brain.step(
                            eye_drive=current_stimulus
                        )

                        window_activity[
                            fired
                        ] += 1

                    window_features.append(
                        window_activity
                    )

                features = np.concatenate(
                    window_features
                )

                X.append(features)

                y.append(
                    position_to_label[
                        position
                    ]
                )


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
    len(position_names)
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
print("5-FOLD POSITION GENERALIZATION")
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
            range(len(position_names))
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
    all_predictions
    == all_targets
)

print("\n" + "=" * 70)
print("FINAL RESULTS")
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
    "\nChance level: 20.00%"
)


# =========================================================
# PER-POSITION RESULTS
# =========================================================

print("\nPer-position results:")

for class_index, position in enumerate(
    position_names
):

    mask = (
        all_targets
        == class_index
    )

    accuracy = np.mean(
        all_predictions[mask]
        == all_targets[mask]
    )

    print(
        f"{position:15s}: "
        f"{accuracy * 100:.2f}%"
    )


print(
    "\nPosition context generalization complete."
)