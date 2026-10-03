import numpy as np

from flybrain import FlyBrain
from fly_encoder import FlyEncoder
from fly_decoder import FlyDecoder


# ============================================================
# SETTINGS
# ============================================================

TRIALS_PER_COMBINATION = 20
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
# ALL COMBINATIONS
# ============================================================

combinations = []

for position in positions:

    for intensity_name in intensity_names:

        intensity = encoder.intensities[
            intensity_name
        ]

        combinations.append(
            (
                position,
                intensity_name,
                intensity
            )
        )


print()
print("=" * 70)
print("FLY-001 COMPOSITIONAL GENERALIZATION TEST")
print("=" * 70)

print()
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
# COLLECT FRESH DATA
# ============================================================

X = []
position_labels = []
intensity_labels = []
combination_labels = []


print()
print("=" * 70)
print("DATA COLLECTION")
print("=" * 70)
print()


for position, intensity_name, intensity in combinations:

    print(
        f"Collecting "
        f"{position} + {intensity_name}..."
    )

    for trial in range(
        TRIALS_PER_COMBINATION
    ):

        activity = run_trial(
            position,
            intensity
        )

        X.append(activity)

        position_labels.append(
            position
        )

        intensity_labels.append(
            intensity_name
        )

        combination_labels.append(
            f"{position}_{intensity_name}"
        )

        print(
            f"  Trial "
            f"{trial + 1:02d}/"
            f"{TRIALS_PER_COMBINATION}",
            end="\r"
        )

    print()


X = np.asarray(X)

position_labels = np.asarray(
    position_labels
)

intensity_labels = np.asarray(
    intensity_labels
)

combination_labels = np.asarray(
    combination_labels
)


print()
print("Data collection complete.")
print(
    f"Neural dimensions: {X.shape[1]}"
)


# ============================================================
# LEAVE-ONE-COMBINATION-OUT TEST
# ============================================================

position_results = []
intensity_results = []
combined_results = []


print()
print("=" * 70)
print("LEAVE-ONE-COMBINATION-OUT DECODING")
print("=" * 70)
print()


for held_out in combinations:

    held_position = held_out[0]
    held_intensity = held_out[1]

    held_label = (
        f"{held_position}_{held_intensity}"
    )

    test_mask = (
        combination_labels ==
        held_label
    )

    train_mask = ~test_mask

    X_train = X[train_mask]
    X_test = X[test_mask]

    y_position_train = (
        position_labels[train_mask]
    )

    y_position_test = (
        position_labels[test_mask]
    )

    y_intensity_train = (
        intensity_labels[train_mask]
    )

    y_intensity_test = (
        intensity_labels[test_mask]
    )


    # --------------------------------------------------------
    # POSITION DECODER
    # --------------------------------------------------------

    position_decoder = FlyDecoder(
        positions,
        top_features=TOP_FEATURES
    )

    position_decoder.fit(
        X_train,
        y_position_train
    )

    position_prediction = (
        position_decoder.predict(
            X_test
        )
    )

    position_accuracy = np.mean(
        position_prediction ==
        y_position_test
    )


    # --------------------------------------------------------
    # INTENSITY DECODER
    # --------------------------------------------------------

    intensity_decoder = FlyDecoder(
        intensity_names,
        top_features=TOP_FEATURES
    )

    intensity_decoder.fit(
        X_train,
        y_intensity_train
    )

    intensity_prediction = (
        intensity_decoder.predict(
            X_test
        )
    )

    intensity_accuracy = np.mean(
        intensity_prediction ==
        y_intensity_test
    )


    # --------------------------------------------------------
    # COMBINED FACTOR ACCURACY
    # --------------------------------------------------------

    combined_accuracy = np.mean(
        (
            position_prediction ==
            y_position_test
        )
        &
        (
            intensity_prediction ==
            y_intensity_test
        )
    )


    position_results.append(
        position_accuracy
    )

    intensity_results.append(
        intensity_accuracy
    )

    combined_results.append(
        combined_accuracy
    )


    print(
        f"{held_label:28s} | "
        f"Position: "
        f"{position_accuracy * 100:5.1f}% | "
        f"Intensity: "
        f"{intensity_accuracy * 100:5.1f}% | "
        f"Both: "
        f"{combined_accuracy * 100:5.1f}%"
    )


# ============================================================
# SUMMARY
# ============================================================

position_results = np.asarray(
    position_results
)

intensity_results = np.asarray(
    intensity_results
)

combined_results = np.asarray(
    combined_results
)


print()
print("=" * 70)
print("COMPOSITIONAL GENERALIZATION RESULTS")
print("=" * 70)

print()

print(
    f"Position accuracy: "
    f"{position_results.mean() * 100:.2f}%"
)

print(
    "Position chance: 20.00%"
)

print()

print(
    f"Intensity accuracy: "
    f"{intensity_results.mean() * 100:.2f}%"
)

print(
    "Intensity chance: 25.00%"
)

print()

print(
    f"Both position + intensity: "
    f"{combined_results.mean() * 100:.2f}%"
)

print(
    "Exact combination chance: "
    f"5.00%"
)

print()

print("=" * 70)
print("COMPOSITIONAL TEST COMPLETE")
print("=" * 70)