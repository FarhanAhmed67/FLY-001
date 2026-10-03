import numpy as np
from flybrain import FlyBrain


# ============================================================
# SETTINGS
# ============================================================

SIGMA = 0.08
BRAIN_STEPS = 50
MOVE_SCALE = 5.0
MAX_WORLD_X = 100.0


# ============================================================
# SOFT VISUAL ENCODER
# ============================================================

def soft_encode(azimuth, position, sigma=SIGMA):

    azimuth = np.asarray(
        azimuth,
        dtype=np.float32
    )

    distance = np.abs(
        azimuth - position
    )

    stimulus = np.exp(
        -(distance ** 2)
        / (2.0 * sigma ** 2)
    )

    stimulus[~np.isfinite(azimuth)] = 0.0

    maximum = stimulus.max()

    if maximum > 0:
        stimulus /= maximum

    return stimulus.astype(
        np.float32
    )


# ============================================================
# LOAD CONTINUOUS DECODER
# ============================================================

print()
print("=" * 65)
print("FLY-001 CONTINUOUS CLOSED-LOOP TEST")
print("=" * 65)

print()
print("Loading FlyBrain...")

brain = FlyBrain(
    device="auto"
)

print(
    f"Neurons          : {len(brain.cell_type)}"
)

print(
    f"Visual receptors : {len(brain.azimuth)}"
)


print()
print("Loading continuous decoder...")

model = np.load(
    "fly_soft_continuous_decoder.npz"
)

features = model["features"]
mean = model["mean"]
std = model["std"]
weights = model["weights"]

print(
    f"Decoder features : {len(features)}"
)


# ============================================================
# WORLD
# ============================================================

fly_x = 50.0
fly_y = 50.0

light_x = 80.0
light_y = 65.0


# ============================================================
# HELPERS
# ============================================================

def get_horizontal_target():

    dx = light_x - fly_x

    # Convert world distance into
    # normalized visual coordinate.

    position = dx / 40.0

    position = np.clip(
        position,
        -1.0,
        1.0
    )

    return position


def decode_brain(stimulus):

    brain.reset()

    spike_counts = np.zeros(
        len(brain.cell_type),
        dtype=np.float32
    )

    for _ in range(BRAIN_STEPS):

        fired = brain.step(
            eye_drive=stimulus
        )

        if len(fired) > 0:

            spike_counts += np.bincount(
                fired,
                minlength=len(brain.cell_type)
            )

    selected = spike_counts[
        features
    ]

    normalized = (
        selected - mean
    ) / std

    prediction = (
        normalized @ weights
    )

    return (
        float(prediction),
        int(spike_counts.sum())
    )


# ============================================================
# CLOSED LOOP
# ============================================================

print()
print("=" * 65)
print("WORLD INITIALIZED")
print("=" * 65)

print(
    f"Fly position   : ({fly_x:.1f}, {fly_y:.1f})"
)

print(
    f"Light position : ({light_x:.1f}, {light_y:.1f})"
)

print()
print("=" * 65)
print("STARTING CLOSED LOOP")
print("=" * 65)


for step in range(1, 21):

    # --------------------------------------------------------
    # WORLD → SENSOR
    # --------------------------------------------------------

    true_position = get_horizontal_target()

    stimulus = soft_encode(
        brain.azimuth,
        true_position
    )

    # --------------------------------------------------------
    # SENSOR → BRAIN → DECODER
    # --------------------------------------------------------

    decoded_position, spikes = decode_brain(
        stimulus
    )

    # --------------------------------------------------------
    # BRAIN → ACTION
    # --------------------------------------------------------

    movement = (
        decoded_position
        * MOVE_SCALE
    )

    fly_x += movement

    fly_x = np.clip(
        fly_x,
        0.0,
        MAX_WORLD_X
    )

    # --------------------------------------------------------
    # WORLD STATE
    # --------------------------------------------------------

    dx = light_x - fly_x

    distance = np.sqrt(
        dx ** 2
        + (light_y - fly_y) ** 2
    )

    print()
    print(
        f"STEP {step:02d}"
    )

    print(
        f"  True visual position : "
        f"{true_position:+.3f}"
    )

    print(
        f"  Brain decoded        : "
        f"{decoded_position:+.3f}"
    )

    print(
        f"  Movement             : "
        f"{movement:+.2f}"
    )

    print(
        f"  Neural spikes        : "
        f"{spikes:,}"
    )

    print(
        f"  Fly position         : "
        f"({fly_x:.2f}, {fly_y:.2f})"
    )

    print(
        f"  Distance to light    : "
        f"{distance:.2f}"
    )


print()
print("=" * 65)
print("CLOSED LOOP COMPLETE")
print("=" * 65)

print()
print(
    f"Final fly position : "
    f"({fly_x:.2f}, {fly_y:.2f})"
)

print(
    f"Light position     : "
    f"({light_x:.2f}, {light_y:.2f})"
)

print(
    f"Final distance     : "
    f"{distance:.2f}"
)

print()