import numpy as np
from flybrain import FlyBrain

SIGMA = 0.08
BRAIN_STEPS = 50
SEED = 42


def soft_signal(azimuth, position, sigma=SIGMA):
    azimuth = np.asarray(azimuth, dtype=np.float32)

    distance = np.abs(azimuth - position)

    signal = np.exp(
        -(distance ** 2) / (2.0 * sigma ** 2)
    )

    signal[~np.isfinite(azimuth)] = 0.0

    maximum = signal.max()

    if maximum > 0:
        signal /= maximum

    return signal.astype(np.float32)


def make_2d_input(azimuth, x_position, y_position):
    n = len(azimuth)
    half = n // 2

    horizontal = soft_signal(
        azimuth,
        x_position
    )

    vertical = soft_signal(
        azimuth,
        y_position
    )

    stimulus = np.zeros(
        n,
        dtype=np.float32
    )

    # Project-level 2D sensory split
    stimulus[:half] = horizontal[:half]
    stimulus[half:] = vertical[half:]

    return stimulus


def distance(x1, y1, x2, y2):
    return np.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


def world_to_target(
    fly_x,
    fly_y,
    light_x,
    light_y
):
    dx = light_x - fly_x
    dy = light_y - fly_y

    target_x = np.clip(
        dx / 40.0,
        -1.0,
        1.0
    )

    target_y = np.clip(
        dy / 40.0,
        -1.0,
        1.0
    )

    return target_x, target_y


def predict(
    activity,
    features,
    coef,
    intercept
):
    prediction = float(
        activity[features] @ coef + intercept
    )

    return float(
        np.clip(
            prediction,
            -1.0,
            1.0
        )
    )


print("=" * 70)
print("FLY-001 2D NAVIGATION DIAGNOSTIC")
print("=" * 70)


# ---------------------------------------------------------
# LOAD BRAIN
# ---------------------------------------------------------

brain = FlyBrain(
    device="auto"
)


# ---------------------------------------------------------
# LOAD TRAINED DECODER
# ---------------------------------------------------------

model = np.load(
    "fly_2d_continuous_decoder.npz"
)

features_x = model["features_x"]
features_y = model["features_y"]

coef_x = model["coef_x"]
coef_y = model["coef_y"]

intercept_x = float(
    model["intercept_x"]
)

intercept_y = float(
    model["intercept_y"]
)


# ---------------------------------------------------------
# CREATE WORLD
# ---------------------------------------------------------

rng = np.random.default_rng(
    SEED
)

while True:

    fly_x = rng.uniform(
        10,
        90
    )

    fly_y = rng.uniform(
        10,
        90
    )

    light_x = rng.uniform(
        10,
        90
    )

    light_y = rng.uniform(
        10,
        90
    )

    initial_distance = distance(
        fly_x,
        fly_y,
        light_x,
        light_y
    )

    if initial_distance >= 20:
        break


print()

print(
    f"Initial fly   : "
    f"({fly_x:.2f}, {fly_y:.2f})"
)

print(
    f"Target        : "
    f"({light_x:.2f}, {light_y:.2f})"
)

print(
    f"Initial dist  : "
    f"{initial_distance:.2f}"
)

print()

print("=" * 70)
print("STEP TRACE")
print("=" * 70)


# ---------------------------------------------------------
# NAVIGATION LOOP
# ---------------------------------------------------------

for step in range(1, 51):

    current_distance = distance(
        fly_x,
        fly_y,
        light_x,
        light_y
    )


    # -----------------------------------------------------
    # CHECK IF TARGET REACHED
    # -----------------------------------------------------

    if current_distance < 1.5:

        print()
        print(
            f"TARGET REACHED at step {step - 1}"
        )

        break


    # -----------------------------------------------------
    # CONVERT WORLD TARGET TO SENSORY TARGET
    # -----------------------------------------------------

    target_x, target_y = world_to_target(
        fly_x,
        fly_y,
        light_x,
        light_y
    )


    # -----------------------------------------------------
    # RESET BRAIN
    # -----------------------------------------------------

    brain.reset()


    # -----------------------------------------------------
    # CREATE 2D VISUAL INPUT
    # -----------------------------------------------------

    stimulus = make_2d_input(
        brain.azimuth,
        target_x,
        target_y
    )


    # -----------------------------------------------------
    # RUN FLY BRAIN
    # -----------------------------------------------------

    activity = np.zeros(
        len(brain.cell_type),
        dtype=np.float32
    )


    for _ in range(BRAIN_STEPS):

        fired = brain.step(
            eye_drive=stimulus
        )

        if len(fired) > 0:

            activity[fired] += 1


    # -----------------------------------------------------
    # DECODE X / Y
    # -----------------------------------------------------

    decoded_x = predict(
        activity,
        features_x,
        coef_x,
        intercept_x
    )

    decoded_y = predict(
        activity,
        features_y,
        coef_y,
        intercept_y
    )


    # -----------------------------------------------------
    # NORMALIZE DECODER VECTOR
    # -----------------------------------------------------

    vector = np.array(
        [
            decoded_x,
            decoded_y
        ],
        dtype=np.float32
    )

    magnitude = np.linalg.norm(
        vector
    )


    if magnitude > 1e-6:

        direction = (
            vector / magnitude
        )

    else:

        direction = np.zeros(
            2,
            dtype=np.float32
        )


    # -----------------------------------------------------
    # ADAPTIVE MOVEMENT SCALE
    # -----------------------------------------------------

    if current_distance > 12:

        scale = 4.0

    elif current_distance > 6:

        scale = 3.0

    elif current_distance > 3:

        scale = 2.0

    else:

        scale = 1.0


    # -----------------------------------------------------
    # CALCULATE MOVEMENT
    # -----------------------------------------------------

    move_x = float(
        direction[0] * scale
    )

    move_y = float(
        direction[1] * scale
    )


    # -----------------------------------------------------
    # PREVENT OVERSHOOT
    # -----------------------------------------------------

    proposed_x = fly_x + move_x
    proposed_y = fly_y + move_y

    proposed_distance = distance(
        proposed_x,
        proposed_y,
        light_x,
        light_y
    )


    if proposed_distance > current_distance:

        # Reduce movement if it would
        # make the fly farther from target.

        move_x *= 0.5
        move_y *= 0.5


    # -----------------------------------------------------
    # PRINT STEP
    # -----------------------------------------------------

    print(
        f"{step:02d} | "
        f"dist={current_distance:6.2f} | "
        f"target=({target_x:+.3f},{target_y:+.3f}) | "
        f"decoded=({decoded_x:+.3f},{decoded_y:+.3f}) | "
        f"move=({move_x:+.2f},{move_y:+.2f})"
    )


    # -----------------------------------------------------
    # APPLY MOVEMENT
    # -----------------------------------------------------

    fly_x += move_x
    fly_y += move_y


    # -----------------------------------------------------
    # WORLD BOUNDARIES
    # -----------------------------------------------------

    fly_x = np.clip(
        fly_x,
        0,
        100
    )

    fly_y = np.clip(
        fly_y,
        0,
        100
    )


# ---------------------------------------------------------
# FINAL RESULT
# ---------------------------------------------------------

print()

print("=" * 70)

final_distance = distance(
    fly_x,
    fly_y,
    light_x,
    light_y
)

print(
    f"Final fly    : "
    f"({fly_x:.2f}, {fly_y:.2f})"
)

print(
    f"Target       : "
    f"({light_x:.2f}, {light_y:.2f})"
)

print(
    f"Final dist   : "
    f"{final_distance:.2f}"
)

print(
    f"Distance reduction : "
    f"{((initial_distance - final_distance) / initial_distance) * 100:.2f}%"
)

if final_distance < 1.5:

    print(
        "STATUS       : TARGET REACHED"
    )

else:

    print(
        "STATUS       : TARGET NOT REACHED"
    )

print("=" * 70)