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

WINDOW_SIZE = 10
N_WINDOWS = 5
N_RECEPTORS = 500


# =========================================================
# EXPERIMENT DESIGN
# =========================================================
#
# Each position has 12 possible contexts:
#
#   3 intensities × 4 temporal patterns
#
# We hold out 4 contexts per position.
#
# The SAME context combinations are held out for every
# position, making the test balanced.
#
# The decoder therefore never sees those exact
# position × intensity × temporal combinations during
# training.
#
# =========================================================

HELD_OUT_CONTEXTS = {
    ("DIM", "STATIC"),
    ("MEDIUM", "PULSE"),
    ("BRIGHT", "DOUBLE_PULSE"),
    ("DIM", "SUSTAINED"),
}


# =========================================================
# LABELS
# =========================================================

position_names = list(
    POSITIONS.keys()
)

position_to_label = {
    name: index
    for index, name in enumerate(position_names)
}


print("=" * 70)
print("FLY-001 HELD-OUT CONTEXT GENERALIZATION")
print("=" * 70)

print(
    "Task: decode POSITION"
)

print(
    "Training contexts: 8 per position"
)

print(
    "Held-out contexts: 4 per position"
)

print(
    f"Trials per context: {TRIALS_PER_CONTEXT}"
)

print(
    "\nHeld-out contexts:"
)

for intensity, temporal in HELD_OUT_CONTEXTS:
    print(
        f"  {intensity:6s} + {temporal}"
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
# COLLECT TRAINING AND TEST DATA
# =========================================================

X_train = []
y_train = []

X_test = []
y_test = []

train_context_count = 0
test_context_count = 0


print("\n" + "=" * 70)
print("COLLECTING TRAINING DATA")
print("=" * 70)


for position in POSITIONS:

    for intensity in INTENSITIES:

        for temporal in TEMPORAL_PATTERNS:

            context = (
                intensity,
                temporal
            )

            is_test_context = (
                context
                in HELD_OUT_CONTEXTS
            )

            pattern = TEMPORAL_PATTERNS[
                temporal
            ]

            stimulus = (
                position_masks[position]
                * INTENSITIES[intensity]
            )

            if is_test_context:
                destination_X = X_test
                destination_y = y_test

                print(
                    f"TEST   {position:12s} + "
                    f"{intensity:6s} + "
                    f"{temporal:13s}"
                )

                test_context_count += 1

            else:
                destination_X = X_train
                destination_y = y_train

                print(
                    f"TRAIN  {position:12s} + "
                    f"{intensity:6s} + "
                    f"{temporal:13s}"
                )

                train_context_count += 1


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

                destination_X.append(
                    features
                )

                destination_y.append(
                    position_to_label[
                        position
                    ]
                )


X_train = np.asarray(
    X_train,
    dtype=np.float32
)

y_train = np.asarray(
    y_train
)

X_test = np.asarray(
    X_test,
    dtype=np.float32
)

y_test = np.asarray(
    y_test
)


# =========================================================
# DATA SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("DATA COLLECTION COMPLETE")
print("=" * 70)

print(
    f"Training contexts: "
    f"{train_context_count}"
)

print(
    f"Testing contexts: "
    f"{test_context_count}"
)

print(
    f"Training samples: "
    f"{len(X_train)}"
)

print(
    f"Testing samples: "
    f"{len(X_test)}"
)

print(
    f"Feature matrix: "
    f"{X_train.shape}"
)

print(
    f"Test matrix: "
    f"{X_test.shape}"
)


# =========================================================
# TRAIN DECODER
# =========================================================

print("\n" + "=" * 70)
print("TRAINING POSITION DECODER")
print("=" * 70)

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

print(
    "Decoder trained."
)


# =========================================================
# TEST
# =========================================================

print("\n" + "=" * 70)
print("HELD-OUT CONTEXT TEST")
print("=" * 70)

predictions = decoder.predict(
    X_test
)

accuracy = np.mean(
    predictions == y_test
)

print(
    f"\nOverall accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    "Chance level: 20.00%"
)


# =========================================================
# PER-POSITION RESULTS
# =========================================================

print("\nPer-position results:")

for class_index, position in enumerate(
    position_names
):

    mask = (
        y_test
        == class_index
    )

    position_accuracy = np.mean(
        predictions[mask]
        == y_test[mask]
    )

    print(
        f"{position:15s}: "
        f"{position_accuracy * 100:.2f}%"
    )


# =========================================================
# PER-CONTEXT RESULTS
# =========================================================

print("\nPer-held-out-context results:")

test_offset = 0

for position in POSITIONS:

    for intensity, temporal in (
        HELD_OUT_CONTEXTS
    ):

        start = test_offset

        end = (
            start
            + TRIALS_PER_CONTEXT
        )

        context_predictions = (
            predictions[start:end]
        )

        context_targets = (
            y_test[start:end]
        )

        context_accuracy = np.mean(
            context_predictions
            == context_targets
        )

        print(
            f"{position:12s} + "
            f"{intensity:6s} + "
            f"{temporal:13s}: "
            f"{context_accuracy * 100:.2f}%"
        )

        test_offset = end


# =========================================================
# FINAL INTERPRETATION
# =========================================================

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)

if accuracy >= 0.80:

    print(
        "Strong held-out position "
        "generalization."
    )

elif accuracy >= 0.50:

    print(
        "Moderate held-out position "
        "generalization."
    )

else:

    print(
        "Weak held-out position "
        "generalization."
    )

print(
    "\nImportant:"
)

print(
    "The decoder was never trained on the "
    "four held-out intensity/temporal contexts."
)

print(
    "This is therefore a stronger test than "
    "randomly mixing all contexts."
)