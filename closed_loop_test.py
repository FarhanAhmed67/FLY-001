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

classes = list(
    encoder.concepts.keys()
)

decoder = FlyDecoder(
    classes,
    top_features=1000
)


# ============================================================
# RUN ONE TRIAL
# ============================================================

def run_trial(concept):

    brain.reset()

    stimulus = encoder.encode(
        concept
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
y = []


print()
print("=" * 70)
print("FLY-001 CLOSED LOOP DATA COLLECTION")
print("=" * 70)
print()

for concept in classes:

    print(
        f"Collecting {concept}..."
    )

    for trial in range(
        TRIALS_PER_CLASS
    ):

        activity = run_trial(
            concept
        )

        X.append(activity)
        y.append(concept)

        print(
            f"  Trial "
            f"{trial + 1:02d}/"
            f"{TRIALS_PER_CLASS}",
            end="\r"
        )

    print()


X = np.asarray(X)
y = np.asarray(y)


print()
print(
    f"Collected {len(y)} trials."
)

print(
    f"Neural dimensions: {X.shape[1]}"
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

indices = np.arange(
    len(y)
)

rng.shuffle(indices)

train_indices = []
test_indices = []


# Keep equal class representation.
for cls in classes:

    cls_indices = indices[
        y[indices] == cls
    ]

    split = int(
        len(cls_indices) * 0.8
    )

    train_indices.extend(
        cls_indices[:split]
    )

    test_indices.extend(
        cls_indices[split:]
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

y_train = y[
    train_indices
]

X_test = X[
    test_indices
]

y_test = y[
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
# TRAIN DECODER
# ============================================================

print()
print("=" * 70)
print("TRAINING BRAIN → CONCEPT DECODER")
print("=" * 70)

decoder.fit(
    X_train,
    y_train
)


# ============================================================
# TEST
# ============================================================

predictions = decoder.predict(
    X_test
)


accuracy = np.mean(
    predictions == y_test
)


print()
print("=" * 70)
print("DECODER RESULTS")
print("=" * 70)

print()

for actual, predicted in zip(
    y_test,
    predictions
):

    status = (
        "✓"
        if actual == predicted
        else "✗"
    )

    print(
        f"{status} "
        f"Actual: {actual:12s} "
        f"Predicted: {predicted}"
    )


print()
print(
    f"Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    "Chance level: 20.00%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

matrix = np.zeros(
    (len(classes), len(classes)),
    dtype=int
)

class_to_index = {
    cls: i
    for i, cls in enumerate(classes)
}

for actual, predicted in zip(
    y_test,
    predictions
):

    matrix[
        class_to_index[actual],
        class_to_index[predicted]
    ] += 1


print()

print(
    "Actual \\ Predicted"
)

print(
    "              "
    + " ".join(
        f"{cls[:6]:>7s}"
        for cls in classes
    )
)

for i, cls in enumerate(classes):

    print(
        f"{cls[:14]:14s}"
        + " ".join(
            f"{value:7d}"
            for value in matrix[i]
        )
    )


print()
print("=" * 70)
print("CLOSED LOOP TEST COMPLETE")
print("=" * 70)