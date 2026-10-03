import numpy as np

from flybrain import FlyBrain
from fly_encoder import FlyEncoder
from fly_decoder import FlyDecoder


# ============================================================
# SETTINGS
# ============================================================

TRIALS_PER_CONDITION = 20
STEPS = 50
TOP_FEATURES = 1000

rng = np.random.default_rng(42)


# ============================================================
# INITIALIZE
# ============================================================

brain = FlyBrain(device="auto")

encoder = FlyEncoder(
    brain.azimuth
)

positions = list(
    encoder.positions.keys()
)

intensity_names = list(
    encoder.intensities.keys()
)


# ============================================================
# RUN ONE TRIAL
# ============================================================

def run_trial(position, intensity_name):

    intensity = encoder.intensities[
        intensity_name
    ]

    stimulus = encoder.encode_position(
        position,
        intensity
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
                minlength=len(brain.cell_type)
            )[:len(brain.cell_type)]

    return activity


# ============================================================
# DATA COLLECTION
# ============================================================

def collect_data(position_list):

    X = []
    y = []
    position_labels = []

    print()

    for position in position_list:

        print(
            f"Collecting {position}..."
        )

        for intensity_name in intensity_names:

            print(
                f"  {intensity_name}: ",
                end=""
            )

            for trial in range(
                TRIALS_PER_CONDITION
            ):

                activity = run_trial(
                    position,
                    intensity_name
                )

                X.append(activity)

                y.append(
                    intensity_name
                )

                position_labels.append(
                    position
                )

                print(
                    ".",
                    end="",
                    flush=True
                )

            print()

    return (
        np.asarray(X),
        np.asarray(y),
        np.asarray(position_labels)
    )


# ============================================================
# EXPERIMENT A
# FIXED POSITION
# ============================================================

print()
print("=" * 70)
print("EXPERIMENT A — INTENSITY ONLY")
print("=" * 70)

print()
print(
    "Position: CENTER"
)

print(
    "4 intensity classes"
)

print(
    f"{TRIALS_PER_CONDITION} trials per class"
)


X_a, y_a, positions_a = collect_data(
    ["CENTER"]
)


# ------------------------------------------------------------
# 5-FOLD CROSS VALIDATION
# ------------------------------------------------------------

indices = np.arange(
    len(X_a)
)

rng.shuffle(indices)

folds = np.array_split(
    indices,
    5
)

results_a = []


for fold_index in range(5):

    test_idx = folds[fold_index]

    train_idx = np.concatenate(
        [
            folds[i]
            for i in range(5)
            if i != fold_index
        ]
    )

    decoder = FlyDecoder(
        intensity_names,
        top_features=TOP_FEATURES
    )

    decoder.fit(
        X_a[train_idx],
        y_a[train_idx]
    )

    prediction = decoder.predict(
        X_a[test_idx]
    )

    accuracy = np.mean(
        prediction ==
        y_a[test_idx]
    )

    results_a.append(
        accuracy
    )

    print(
        f"Fold {fold_index + 1}: "
        f"{accuracy * 100:.2f}%"
    )


accuracy_a = np.mean(
    results_a
)


print()
print(
    f"Experiment A accuracy: "
    f"{accuracy_a * 100:.2f}%"
)

print(
    "Chance: 25.00%"
)


# ============================================================
# EXPERIMENT B
# CROSS-POSITION GENERALIZATION
# ============================================================

print()
print("=" * 70)
print("EXPERIMENT B — CROSS-POSITION INTENSITY GENERALIZATION")
print("=" * 70)

print()

X_b, y_b, positions_b = collect_data(
    positions
)


results_b = []


# ============================================================
# LEAVE-ONE-POSITION-OUT
# ============================================================

for held_out_position in positions:

    print()
    print(
        "-" * 70
    )

    print(
        f"HOLDING OUT: "
        f"{held_out_position}"
    )

    print(
        "-" * 70
    )


    test_mask = (
        positions_b ==
        held_out_position
    )

    train_mask = ~test_mask


    X_train = X_b[
        train_mask
    ]

    X_test = X_b[
        test_mask
    ]

    y_train = y_b[
        train_mask
    ]

    y_test = y_b[
        test_mask
    ]


    decoder = FlyDecoder(
        intensity_names,
        top_features=TOP_FEATURES
    )


    decoder.fit(
        X_train,
        y_train
    )


    prediction = decoder.predict(
        X_test
    )


    accuracy = np.mean(
        prediction ==
        y_test
    )


    results_b.append(
        accuracy
    )


    print()

    print(
        f"Held-out position: "
        f"{held_out_position}"
    )

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        "Chance: 25.00%"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

accuracy_b = np.mean(
    results_b
)


print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print()

print(
    "Experiment A:"
)

print(
    f"Fixed CENTER position: "
    f"{accuracy_a * 100:.2f}%"
)

print(
    "Chance: 25.00%"
)


print()

print(
    "Experiment B:"
)

print(
    f"Cross-position average: "
    f"{accuracy_b * 100:.2f}%"
)

print(
    "Chance: 25.00%"
)


print()

print(
    "Per-position cross-generalization:"
)

for position, accuracy in zip(
    positions,
    results_b
):

    print(
        f"  {position:15s}: "
        f"{accuracy * 100:.2f}%"
    )


print()
print("=" * 70)
print("INTENSITY GENERALIZATION TEST COMPLETE")
print("=" * 70)