import numpy as np

from flybrain import FlyBrain

from semantic_encoder import (
    SemanticEncoder
)

from fly_decoder import FlyDecoder


# ============================================================
# SETTINGS
# ============================================================

TRIALS_PER_CONCEPT = 20
STEPS = 50
TOP_FEATURES = 1000

rng = np.random.default_rng(42)


# ============================================================
# INITIALIZE
# ============================================================

brain = FlyBrain(
    device="auto"
)

encoder = SemanticEncoder(
    brain.azimuth
)

concepts = encoder.get_concepts()


print()
print("=" * 70)
print("FLY-001 SEMANTIC CONCEPT TEST")
print("=" * 70)

print()
print(
    f"Concepts: {len(concepts)}"
)

print(
    f"Trials per concept: "
    f"{TRIALS_PER_CONCEPT}"
)

print(
    f"Total trials: "
    f"{len(concepts) * TRIALS_PER_CONCEPT}"
)


# ============================================================
# RUN TRIAL
# ============================================================

def run_trial(concept):

    stimulus = encoder.encode(
        concept
    )

    brain.reset()

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
                minlength=len(
                    brain.cell_type
                )
            )[
                :len(brain.cell_type)
            ]

    return activity


# ============================================================
# COLLECT DATA
# ============================================================

X = []
y = []


print()
print("=" * 70)
print("DATA COLLECTION")
print("=" * 70)
print()


for concept in concepts:

    print(
        f"Collecting {concept}..."
    )

    for trial in range(
        TRIALS_PER_CONCEPT
    ):

        activity = run_trial(
            concept
        )

        X.append(
            activity
        )

        y.append(
            concept
        )

        print(
            f"  Trial "
            f"{trial + 1:02d}/"
            f"{TRIALS_PER_CONCEPT}",
            end="\r"
        )

    print()


X = np.asarray(X)

y = np.asarray(y)


print()
print("Data collection complete.")

print(
    f"Neural matrix: {X.shape}"
)


# ============================================================
# SHUFFLED 80/20 SPLIT
# ============================================================

indices = np.arange(
    len(X)
)

rng.shuffle(
    indices
)

split = int(
    len(X) * 0.80
)

train_idx = indices[
    :split
]

test_idx = indices[
    split:
]


# ============================================================
# TRAIN DECODER
# ============================================================

decoder = FlyDecoder(
    concepts,
    top_features=TOP_FEATURES
)

decoder.fit(
    X[train_idx],
    y[train_idx]
)


# ============================================================
# TEST
# ============================================================

prediction = decoder.predict(
    X[test_idx]
)

accuracy = np.mean(
    prediction ==
    y[test_idx]
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("SEMANTIC DECODING RESULTS")
print("=" * 70)

print()

print(
    f"Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Chance: "
    f"{100 / len(concepts):.2f}%"
)


# ============================================================
# PER-CONCEPT RESULTS
# ============================================================

print()
print("Per-concept results:")
print()


for concept in concepts:

    mask = (
        y[test_idx] ==
        concept
    )

    if np.sum(mask) == 0:
        continue

    concept_accuracy = np.mean(
        prediction[mask] ==
        y[test_idx][mask]
    )

    print(
        f"{concept:10s}: "
        f"{concept_accuracy * 100:6.2f}%"
    )


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

print()

for i in range(
    min(20, len(test_idx))
):

    print(
        f"Expected: "
        f"{y[test_idx[i]]:10s}"
        f" | Predicted: "
        f"{prediction[i]}"
    )


print()
print("=" * 70)
print("SEMANTIC TEST COMPLETE")
print("=" * 70)