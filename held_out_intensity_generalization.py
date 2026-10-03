import numpy as np
from flybrain import FlyBrain
from fly_decoder import FlyDecoder


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
    "STATIC": lambda t: True,
    "PULSE": lambda t: 0 <= t < 10 or 20 <= t < 30,
    "DOUBLE_PULSE": lambda t: 0 <= t < 5 or 15 <= t < 20,
    "SUSTAINED": lambda t: 0 <= t < 25,
}

TRIALS = 5
STEPS = 50
N_RECEPTORS = 500

# Entire position x temporal context combinations held out.
# Every intensity still appears during training.
HELD_OUT_CONTEXTS = {
    ("LEFT", "STATIC"),
    ("CENTER_LEFT", "SUSTAINED"),
    ("CENTER", "PULSE"),
    ("CENTER_RIGHT", "DOUBLE_PULSE"),
    ("RIGHT", "STATIC"),
}


def make_stimulus(azimuth, position, intensity, temporal_pattern, step):

    direction = POSITIONS[position]

    distance = np.abs(azimuth - direction)

    valid = np.where(np.isfinite(distance))[0]

    selected = valid[
        np.argsort(distance[valid])[:N_RECEPTORS]
    ]

    stimulus = np.zeros(
        len(azimuth),
        dtype=np.float32
    )

    active = TEMPORAL_PATTERNS[temporal_pattern](step)

    if active:
        stimulus[selected] = INTENSITIES[intensity]

    return stimulus


def collect_sample(brain, azimuth, position, intensity, temporal_pattern):

    brain.reset()

    windows = []

    for start in range(0, STEPS, 10):

        window_activity = np.zeros(
            brain.n,
            dtype=np.float32
        )

        for step in range(start, start + 10):

            stimulus = make_stimulus(
                azimuth,
                position,
                intensity,
                temporal_pattern,
                step
            )

            fired = brain.step(
                eye_drive=stimulus
            )

            if len(fired) > 0:
                window_activity[fired] += 1

        windows.append(window_activity)

    return np.concatenate(windows)


print("=" * 60)
print("STRICT HELD-OUT INTENSITY GENERALIZATION")
print("=" * 60)

brain = FlyBrain(device="auto")

azimuth = np.asarray(
    brain.azimuth
)

print(f"Neurons: {brain.n}")
print(f"Visual receptors: {len(azimuth)}")
print()

X_train = []
y_train = []

X_test = []
y_test = []

train_contexts = 0
test_contexts = 0

for position in POSITIONS:

    for temporal_pattern in TEMPORAL_PATTERNS:

        context = (
            position,
            temporal_pattern
        )

        is_held_out = context in HELD_OUT_CONTEXTS

        if is_held_out:
            test_contexts += 1
        else:
            train_contexts += 1

        for intensity_index, intensity in enumerate(INTENSITIES):

            for trial in range(TRIALS):

                X = collect_sample(
                    brain,
                    azimuth,
                    position,
                    intensity,
                    temporal_pattern
                )

                if is_held_out:
                    X_test.append(X)
                    y_test.append(intensity_index)
                else:
                    X_train.append(X)
                    y_train.append(intensity_index)

                print(
                    f"{position:12s} "
                    f"{intensity:8s} "
                    f"{temporal_pattern:13s} "
                    f"trial {trial + 1}/{TRIALS} "
                    f"{'TEST' if is_held_out else 'TRAIN'}"
                )

X_train = np.asarray(X_train, dtype=np.float32)
X_test = np.asarray(X_test, dtype=np.float32)

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

print()
print("Training shape:", X_train.shape)
print("Testing shape :", X_test.shape)
print()

decoder = FlyDecoder(
    classes=list(range(len(INTENSITIES))),
    top_features=1000
)

decoder.fit(
    X_train,
    y_train
)

predictions = decoder.predict(X_test)

accuracy = np.mean(
    predictions == y_test
)

chance = 1 / len(INTENSITIES)

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)

print(
    f"Overall accuracy: {accuracy * 100:.2f}%"
)

print(
    f"Chance level: {chance * 100:.2f}%"
)

print()

for index, intensity in enumerate(INTENSITIES):

    mask = y_test == index

    if np.any(mask):

        acc = np.mean(
            predictions[mask] == y_test[mask]
        )

        print(
            f"{intensity:8s}: "
            f"{acc * 100:.2f}%"
        )

print()
print("Held-out contexts:")

for context in HELD_OUT_CONTEXTS:
    print(
        f"  {context[0]} + {context[1]}"
    )

print()
print("=" * 60)
print("DONE")
print("=" * 60)