import numpy as np

from flybrain import FlyBrain
from fly_encoder import FlyEncoder
from fly_decoder import FlyDecoder


# ============================================================
# SETTINGS
# ============================================================

TRIALS_PER_PATTERN = 20
STEPS = 50
TOP_FEATURES = 1000

rng = np.random.default_rng(42)


# ============================================================
# INITIALIZE
# ============================================================

brain = FlyBrain(
    device="auto"
)

encoder = FlyEncoder(
    brain.azimuth
)


patterns = [
    "STATIC",
    "PULSE",
    "DOUBLE_PULSE",
    "SUSTAINED"
]


# ============================================================
# CREATE CENTER STIMULUS
# ============================================================

stimulus = encoder.encode_position(
    "CENTER",
    1.0
)


# ============================================================
# TEMPORAL PATTERNS
# ============================================================

def get_stimulus(pattern, step):

    if pattern == "STATIC":

        return stimulus


    if pattern == "PULSE":

        # 0-9 ON
        # 10-19 OFF
        # 20-29 ON
        # 30-49 OFF

        if (
            0 <= step < 10
            or
            20 <= step < 30
        ):

            return stimulus

        return np.zeros_like(
            stimulus
        )


    if pattern == "DOUBLE_PULSE":

        # 0-5 ON
        # 6-14 OFF
        # 15-20 ON
        # rest OFF

        if (
            0 <= step < 6
            or
            15 <= step < 21
        ):

            return stimulus

        return np.zeros_like(
            stimulus
        )


    if pattern == "SUSTAINED":

        # First half ON
        # Second half OFF

        if step < 25:

            return stimulus

        return np.zeros_like(
            stimulus
        )


    raise ValueError(
        f"Unknown pattern: {pattern}"
    )


# ============================================================
# RUN TRIAL
# ============================================================

def run_trial(pattern):

    brain.reset()

    activity = np.zeros(
        len(brain.cell_type),
        dtype=np.float32
    )

    for step in range(STEPS):

        current_stimulus = (
            get_stimulus(
                pattern,
                step
            )
        )

        fired = brain.step(
            eye_drive=current_stimulus
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
# DATA COLLECTION
# ============================================================

X = []
y = []


print()
print("=" * 70)
print("FLY-001 TEMPORAL SEMANTIC TEST")
print("=" * 70)

print()

print(
    f"Patterns: {len(patterns)}"
)

print(
    f"Trials per pattern: "
    f"{TRIALS_PER_PATTERN}"
)

print(
    f"Total trials: "
    f"{len(patterns) * TRIALS_PER_PATTERN}"
)

print()


for pattern in patterns:

    print(
        f"Collecting {pattern}..."
    )

    for trial in range(
        TRIALS_PER_PATTERN
    ):

        activity = run_trial(
            pattern
        )

        X.append(
            activity
        )

        y.append(
            pattern
        )

        print(
            f"  Trial "
            f"{trial + 1:02d}/"
            f"{TRIALS_PER_PATTERN}",
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
# 80/20 SPLIT
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
    patterns,
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
print("TEMPORAL DECODING RESULTS")
print("=" * 70)

print()

print(
    f"Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    "Chance: 25.00%"
)


# ============================================================
# PER-PATTERN RESULTS
# ============================================================

print()
print("Per-pattern results:")
print()


for pattern in patterns:

    mask = (
        y[test_idx] ==
        pattern
    )

    if np.sum(mask) == 0:
        continue

    pattern_accuracy = np.mean(
        prediction[mask] ==
        y[test_idx][mask]
    )

    print(
        f"{pattern:15s}: "
        f"{pattern_accuracy * 100:6.2f}%"
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
        f"{y[test_idx[i]]:15s}"
        f" | Predicted: "
        f"{prediction[i]}"
    )


print()
print("=" * 70)
print("TEMPORAL TEST COMPLETE")
print("=" * 70)