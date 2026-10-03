import numpy as np

from flybrain import FlyBrain
from fly_encoder import FlyEncoder
from fly_decoder import FlyDecoder


# ============================================================
# SETTINGS
# ============================================================

TRIALS_PER_CLASS = 20
STEPS = 50

rng = np.random.default_rng(42)


# ============================================================
# INITIALIZE
# ============================================================

brain = FlyBrain(device="auto")

encoder = FlyEncoder(
    brain.azimuth
)


# ============================================================
# CREATE 12 UNIQUE INPUT COMBINATIONS
# ============================================================

inputs = []

for position in encoder.positions:

    for intensity_name, intensity in encoder.intensities.items():

        inputs.append(
            (
                position,
                intensity_name,
                intensity
            )
        )


print()
print("=" * 70)
print("FLY-001 POSITION + INTENSITY DECODER")
print("=" * 70)
print()

print(
    f"Input combinations: {len(inputs)}"
)

for position, intensity_name, intensity in inputs:

    print(
        f"{position:12s} + "
        f"{intensity_name:6s} "
        f"({intensity:.2f})"
    )


# ============================================================
# RUN ONE TRIAL
# ============================================================

def run_trial(position, intensity):

    brain.reset()

    stimulus = encoder.encode_position(
        position,
        intensity
    )

    activity = np.zeros(
        len(brain.cell_type),
        dtype=np.float32
    )

    for _ in range(STEPS):

        fired = brain.step(
            eye_drive=stimulus
        )

        fired = np.asarray(
            fired,
            dtype=np.int64
        )

        if len(fired) > 0:

            activity += np.bincount(
                fired,
                minlength=len(brain.cell_type)
            )[:len(brain.cell_type)]

    return activity


# ============================================================
# COLLECT DATA
# ============================================================

X = []
labels = []
positions = []
intensities = []


print()
print("=" * 70)
print("DATA COLLECTION")
print("=" * 70)
print()


for position, intensity_name, intensity in inputs:

    print(
        f"Collecting "
        f"{position} + {intensity_name}..."
    )

    for trial in range(
        TRIALS_PER_CLASS
    ):

        activity = run_trial(
            position,
            intensity
        )

        X.append(activity)

        labels.append(
            f"{position}_{intensity_name}"
        )

        positions.append(position)

        intensities.append(
            intensity_name
        )

        print(
            f"  Trial "
            f"{trial + 1:02d}/"
            f"{TRIALS_PER_CLASS}",
            end="\r"
        )

    print()


X = np.asarray(X)
labels = np.asarray(labels)
positions = np.asarray(positions)
intensities = np.asarray(intensities)


print()
print(
    f"Collected {len(labels)} trials."
)

print(
    f"Neural dimensions: {X.shape[1]}"
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

train_indices = []
test_indices = []


for label in np.unique(labels):

    class_indices = np.where(
        labels == label
    )[0]

    shuffled = class_indices.copy()

    rng.shuffle(shuffled)

    split = int(
        len(shuffled) * 0.8
    )

    train_indices.extend(
        shuffled[:split]
    )

    test_indices.extend(
        shuffled[split:]
    )


train_indices = np.asarray(
    train_indices
)

test_indices = np.asarray(
    test_indices
)


X_train = X[
    train_indices
]

X_test = X[
    test_indices
]

y_train = labels[
    train_indices
]

y_test = labels[
    test_indices
]


print()
print(
    f"Training trials: {len(y_train)}"
)

print(
    f"Testing trials:  {len(y_test)}"
)


# ============================================================
# DECODE ALL 12 COMBINATIONS
# ============================================================

print()
print("=" * 70)
print("12-CLASS DECODING")
print("=" * 70)


decoder = FlyDecoder(
    np.unique(labels),
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


print()
print(
    f"Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    "Chance level: "
    f"{100 / len(np.unique(labels)):.2f}%"
)


# ============================================================
# POSITION DECODING
# ============================================================

print()
print("=" * 70)
print("POSITION DECODING")
print("=" * 70)


position_decoder = FlyDecoder(
    list(encoder.positions.keys()),
    top_features=1000
)

position_decoder.fit(
    X_train,
    positions[train_indices]
)

position_predictions = (
    position_decoder.predict(
        X_test
    )
)

position_accuracy = np.mean(
    position_predictions ==
    positions[test_indices]
)


print()
print(
    f"Position accuracy: "
    f"{position_accuracy * 100:.2f}%"
)

print("Position chance: 20.00%")


# ============================================================
# INTENSITY DECODING
# ============================================================

print()
print("=" * 70)
print("INTENSITY DECODING")
print("=" * 70)


intensity_decoder = FlyDecoder(
    list(encoder.intensities.keys()),
    top_features=1000
)

intensity_decoder.fit(
    X_train,
    intensities[train_indices]
)

intensity_predictions = (
    intensity_decoder.predict(
        X_test
    )
)

intensity_accuracy = np.mean(
    intensity_predictions ==
    intensities[test_indices]
)


print()
print(
    f"Intensity accuracy: "
    f"{intensity_accuracy * 100:.2f}%"
)

print("Intensity chance: 25.00%")


# ============================================================
# RESULTS BY INPUT
# ============================================================

print()
print("=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)
print()


for i in range(
    min(30, len(y_test))
):

    print(
        f"Actual: "
        f"{y_test[i]:25s} "
        f"Predicted: "
        f"{predictions[i]}"
    )


print()
print("=" * 70)
print("POSITION + INTENSITY TEST COMPLETE")
print("=" * 70)