import numpy as np

from flybrain import FlyBrain
from fly_encoder import FlyEncoder
from fly_decoder import FlyDecoder


# ============================================================
# SETTINGS
# ============================================================

TRIALS_PER_PATTERN = 20
STEPS = 50
WINDOW_SIZE = 10
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
# BASE STIMULUS
# ============================================================

stimulus = encoder.encode_position(
    "CENTER",
    1.0
)

dark = np.zeros_like(
    stimulus
)


# ============================================================
# TEMPORAL PATTERN
# ============================================================

def get_stimulus(pattern, step):

    if pattern == "STATIC":

        return stimulus


    if pattern == "PULSE":

        if (
            0 <= step < 10
            or
            20 <= step < 30
        ):

            return stimulus

        return dark


    if pattern == "DOUBLE_PULSE":

        if (
            0 <= step < 6
            or
            15 <= step < 21
        ):

            return stimulus

        return dark


    if pattern == "SUSTAINED":

        if step < 25:

            return stimulus

        return dark


    raise ValueError(
        f"Unknown pattern: {pattern}"
    )


# ============================================================
# RUN TRIAL
# ============================================================

def run_trial(pattern):

    brain.reset()

    window_activity = []

    current_activity = np.zeros(
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

            current_activity += np.bincount(
                fired,
                minlength=len(
                    brain.cell_type
                )
            )[
                :len(brain.cell_type)
            ]


        # End of temporal window
        if (
            (step + 1) %
            WINDOW_SIZE
            == 0
        ):

            window_activity.append(
                current_activity
            )

            current_activity = np.zeros(
                len(brain.cell_type),
                dtype=np.float32
            )


    # Shape:
    # 5 windows × 166700 neurons

    return np.concatenate(
        window_activity
    )


# ============================================================
# DATA COLLECTION
# ============================================================

X = []
y = []


print()
print("=" * 70)
print("FLY-001 TEMPORAL FEATURE DECODER")
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
    f"Temporal windows: "
    f"{STEPS // WINDOW_SIZE}"
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
    f"Temporal feature matrix: "
    f"{X.shape}"
)

print(
    f"Features per trial: "
    f"{X.shape[1]}"
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
print("TEMPORAL FEATURE DECODING RESULTS")
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
print("TEMPORAL FEATURE TEST COMPLETE")
print("=" * 70)