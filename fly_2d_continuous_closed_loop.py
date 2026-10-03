import numpy as np
from flybrain import FlyBrain


SIGMA = 0.08
STEPS = 50

WORLD_SIZE = 100.0
MOVE_SCALE = 5.0

LIGHT_X = 80.0
LIGHT_Y = 75.0

START_X = 20.0
START_Y = 20.0

MAX_STEPS = 30


def soft_signal(n, position, sigma=SIGMA):
    """
    Smooth continuous signal across one artificial sensory channel.
    """
    x = np.linspace(-1.0, 1.0, n)

    signal = np.exp(
        -((x - position) ** 2)
        / (2.0 * sigma ** 2)
    )

    maximum = signal.max()

    if maximum > 0:
        signal /= maximum

    return signal.astype(np.float32)


def make_2d_input(n, x_position, y_position):
    """
    Project-level artificial 2D sensory representation.

    First half  = X channel
    Second half = Y channel

    This is an experimental two-channel representation,
    not native biological elevation encoding.
    """

    half = n // 2

    horizontal = soft_signal(
        half,
        x_position
    )

    vertical = soft_signal(
        n - half,
        y_position
    )

    return np.concatenate(
        [horizontal, vertical]
    ).astype(np.float32)


def clamp(value, low=-1.0, high=1.0):
    return max(low, min(high, value))


def relative_position(delta, scale=40.0):
    """
    Convert world-space displacement into
    the normalized sensory coordinate [-1, +1].
    """

    return clamp(delta / scale)


print()
print("=" * 70)
print("FLY-001 2D CONTINUOUS CLOSED LOOP")
print("=" * 70)


# ---------------------------------------------------------
# LOAD BRAIN
# ---------------------------------------------------------

brain = FlyBrain(device="auto")

n_neurons = len(brain.positions)
n_receptors = len(brain.visual)

print()
print(f"Neurons   : {n_neurons}")
print(f"Receptors : {n_receptors}")


# ---------------------------------------------------------
# LOAD TRAINED DECODER
# ---------------------------------------------------------

model = np.load(
    "fly_2d_continuous_decoder.npz"
)

features_x = model["features_x"]
features_y = model["features_y"]

coef_x = model["coef_x"]
intercept_x = float(model["intercept_x"])

coef_y = model["coef_y"]
intercept_y = float(model["intercept_y"])


print()
print("Loaded:")
print("fly_2d_continuous_decoder.npz")

print()
print(f"X features: {len(features_x)}")
print(f"Y features: {len(features_y)}")


# ---------------------------------------------------------
# WORLD
# ---------------------------------------------------------

fly_x = START_X
fly_y = START_Y

light_x = LIGHT_X
light_y = LIGHT_Y


print()
print("=" * 70)
print("WORLD")
print("=" * 70)

print(
    f"Fly start   : ({fly_x:.2f}, {fly_y:.2f})"
)

print(
    f"Light       : ({light_x:.2f}, {light_y:.2f})"
)


# ---------------------------------------------------------
# CLOSED LOOP
# ---------------------------------------------------------

for step in range(1, MAX_STEPS + 1):

    # -----------------------------------------------------
    # 1. WORLD → RELATIVE TARGET
    # -----------------------------------------------------

    dx = light_x - fly_x
    dy = light_y - fly_y

    distance = np.sqrt(
        dx * dx + dy * dy
    )

    target_x = relative_position(dx)
    target_y = relative_position(dy)


    # -----------------------------------------------------
    # 2. TARGET → ARTIFICIAL 2D SENSORY INPUT
    # -----------------------------------------------------

    stimulus = make_2d_input(
        n_receptors,
        target_x,
        target_y
    )


    # -----------------------------------------------------
    # 3. RUN FLY BRAIN
    # -----------------------------------------------------

    brain.reset()

    spike_counts = np.zeros(
        n_neurons,
        dtype=np.float32
    )

    total_spikes = 0

    for _ in range(STEPS):

        fired = brain.step(
            eye_drive=stimulus
        )

        total_spikes += len(fired)

        if len(fired) > 0:
            spike_counts[fired] += 1


    # -----------------------------------------------------
    # 4. NEURAL ACTIVITY → X/Y DECODERS
    # -----------------------------------------------------

    decoded_x = (
        spike_counts[features_x]
        @ coef_x
    ) + intercept_x

    decoded_y = (
        spike_counts[features_y]
        @ coef_y
    ) + intercept_y


    decoded_x = clamp(
        float(decoded_x)
    )

    decoded_y = clamp(
        float(decoded_y)
    )


    # -----------------------------------------------------
    # 5. DECODER → MOVEMENT
    # -----------------------------------------------------

    move_x = decoded_x * MOVE_SCALE
    move_y = decoded_y * MOVE_SCALE


    # -----------------------------------------------------
    # 6. UPDATE WORLD
    # -----------------------------------------------------

    old_x = fly_x
    old_y = fly_y

    fly_x += move_x
    fly_y += move_y

    fly_x = max(
        0.0,
        min(WORLD_SIZE, fly_x)
    )

    fly_y = max(
        0.0,
        min(WORLD_SIZE, fly_y)
    )


    # -----------------------------------------------------
    # 7. NEW DISTANCE
    # -----------------------------------------------------

    new_dx = light_x - fly_x
    new_dy = light_y - fly_y

    new_distance = np.sqrt(
        new_dx * new_dx
        + new_dy * new_dy
    )


    print()
    print(
        f"STEP {step:02d}"
    )
    print("-" * 70)

    print(
        f"Fly position : "
        f"({old_x:6.2f}, {old_y:6.2f})"
    )

    print(
        f"Target       : "
        f"({target_x:+.3f}, {target_y:+.3f})"
    )

    print(
        f"Brain decode : "
        f"({decoded_x:+.3f}, {decoded_y:+.3f})"
    )

    print(
        f"Movement     : "
        f"({move_x:+.2f}, {move_y:+.2f})"
    )

    print(
        f"New position : "
        f"({fly_x:6.2f}, {fly_y:6.2f})"
    )

    print(
        f"Distance     : "
        f"{distance:6.2f} -> {new_distance:6.2f}"
    )

    print(
        f"Spikes       : "
        f"{total_spikes}"
    )


    # -----------------------------------------------------
    # STOP WHEN CLOSE TO LIGHT
    # -----------------------------------------------------

    if new_distance < 3.0:

        print()
        print("=" * 70)
        print("TARGET REACHED")
        print("=" * 70)

        print(
            f"Fly   : ({fly_x:.2f}, {fly_y:.2f})"
        )

        print(
            f"Light : ({light_x:.2f}, {light_y:.2f})"
        )

        print(
            f"Distance : {new_distance:.2f}"
        )

        break


print()
print("=" * 70)
print("FINAL WORLD STATE")
print("=" * 70)

final_dx = light_x - fly_x
final_dy = light_y - fly_y

final_distance = np.sqrt(
    final_dx * final_dx
    + final_dy * final_dy
)

print(
    f"Fly   : ({fly_x:.2f}, {fly_y:.2f})"
)

print(
    f"Light : ({light_x:.2f}, {light_y:.2f})"
)

print(
    f"Distance : {final_distance:.2f}"
)

print()
print("2D CLOSED LOOP COMPLETE")
print("=" * 70)