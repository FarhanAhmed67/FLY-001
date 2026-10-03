import numpy as np
from flybrain import FlyBrain

SIGMA = 0.08
BRAIN_STEPS = 50

TRIALS = 20
MAX_STEPS = 60

TARGET_THRESHOLD = 2.0

SEED = 42

WORLD_MIN = 5.0
WORLD_MAX = 95.0

TARGET_SPEED = 1.0


def soft_signal(azimuth, position, sigma=SIGMA):

    azimuth = np.asarray(
        azimuth,
        dtype=np.float32
    )

    distance = np.abs(
        azimuth - position
    )

    signal = np.exp(
        -(distance ** 2) /
        (2.0 * sigma ** 2)
    )

    signal[~np.isfinite(azimuth)] = 0.0

    maximum = signal.max()

    if maximum > 0:
        signal /= maximum

    return signal.astype(
        np.float32
    )


def make_2d_input(
    azimuth,
    x_position,
    y_position
):

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

    # Engineered project-level 2D sensory split
    stimulus[:half] = horizontal[:half]
    stimulus[half:] = vertical[half:]

    return stimulus


def distance(
    x1,
    y1,
    x2,
    y2
):

    return np.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


def world_to_target(
    fly_x,
    fly_y,
    target_x,
    target_y
):

    dx = target_x - fly_x
    dy = target_y - fly_y

    normalized_x = np.clip(
        dx / 40.0,
        -1.0,
        1.0
    )

    normalized_y = np.clip(
        dy / 40.0,
        -1.0,
        1.0
    )

    return normalized_x, normalized_y


def predict(
    activity,
    features,
    coef,
    intercept
):

    prediction = float(
        activity[features] @ coef +
        intercept
    )

    return float(
        np.clip(
            prediction,
            -1.0,
            1.0
        )
    )


print("=" * 70)
print("FLY-001 2D MOVING-TARGET BENCHMARK")
print("=" * 70)

print()
print(f"Trials          : {TRIALS}")
print(f"Maximum steps   : {MAX_STEPS}")
print(f"Target radius   : {TARGET_THRESHOLD}")
print(f"Target speed    : {TARGET_SPEED}")
print(f"Random seed     : {SEED}")
print()


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
    model["intercept_y"
    ]
)


# ---------------------------------------------------------
# RANDOM GENERATOR
# ---------------------------------------------------------

rng = np.random.default_rng(
    SEED
)


results = []


# ---------------------------------------------------------
# RUN TRIALS
# ---------------------------------------------------------

for trial in range(
    1,
    TRIALS + 1
):

    # -----------------------------------------------------
    # RANDOM STARTING POSITIONS
    # -----------------------------------------------------

    while True:

        fly_x = rng.uniform(
            10,
            90
        )

        fly_y = rng.uniform(
            10,
            90
        )

        target_x = rng.uniform(
            10,
            90
        )

        target_y = rng.uniform(
            10,
            90
        )

        initial_distance = distance(
            fly_x,
            fly_y,
            target_x,
            target_y
        )

        if initial_distance >= 20:
            break


    # -----------------------------------------------------
    # RANDOM TARGET VELOCITY
    # -----------------------------------------------------

    angle = rng.uniform(
        0,
        2.0 * np.pi
    )

    target_vx = (
        np.cos(angle) *
        TARGET_SPEED
    )

    target_vy = (
        np.sin(angle) *
        TARGET_SPEED
    )


    reached = False
    steps_used = MAX_STEPS


    # -----------------------------------------------------
    # NAVIGATION LOOP
    # -----------------------------------------------------

    for step in range(
        1,
        MAX_STEPS + 1
    ):

        current_distance = distance(
            fly_x,
            fly_y,
            target_x,
            target_y
        )


        # -------------------------------------------------
        # TARGET REACHED
        # -------------------------------------------------

        if current_distance < TARGET_THRESHOLD:

            reached = True
            steps_used = step - 1

            break


        # -------------------------------------------------
        # CONVERT TARGET TO SENSORY INPUT
        # -------------------------------------------------

        sensory_x, sensory_y = world_to_target(
            fly_x,
            fly_y,
            target_x,
            target_y
        )


        # -------------------------------------------------
        # RESET BRAIN
        # -------------------------------------------------

        brain.reset()


        # -------------------------------------------------
        # CREATE VISUAL STIMULUS
        # -------------------------------------------------

        stimulus = make_2d_input(
            brain.azimuth,
            sensory_x,
            sensory_y
        )


        # -------------------------------------------------
        # RUN FLY BRAIN
        # -------------------------------------------------

        activity = np.zeros(
            len(brain.cell_type),
            dtype=np.float32
        )


        for _ in range(
            BRAIN_STEPS
        ):

            fired = brain.step(
                eye_drive=stimulus
            )

            if len(fired) > 0:

                activity[fired] += 1


        # -------------------------------------------------
        # DECODE DIRECTION
        # -------------------------------------------------

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


        # -------------------------------------------------
        # NORMALIZE DIRECTION
        # -------------------------------------------------

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


        # -------------------------------------------------
        # ADAPTIVE MOVEMENT
        # -------------------------------------------------

        if current_distance > 12:

            scale = 4.0

        elif current_distance > 6:

            scale = 3.0

        elif current_distance > 3:

            scale = 2.0

        else:

            scale = 1.0


        move_x = float(
            direction[0] * scale
        )

        move_y = float(
            direction[1] * scale
        )


        # -------------------------------------------------
        # OVERSHOOT PROTECTION
        # -------------------------------------------------

        proposed_x = (
            fly_x + move_x
        )

        proposed_y = (
            fly_y + move_y
        )

        proposed_distance = distance(
            proposed_x,
            proposed_y,
            target_x,
            target_y
        )


        if proposed_distance > current_distance:

            move_x *= 0.5
            move_y *= 0.5


        # -------------------------------------------------
        # MOVE FLY
        # -------------------------------------------------

        fly_x += move_x
        fly_y += move_y


        fly_x = np.clip(
            fly_x,
            WORLD_MIN,
            WORLD_MAX
        )

        fly_y = np.clip(
            fly_y,
            WORLD_MIN,
            WORLD_MAX
        )


        # -------------------------------------------------
        # MOVE TARGET
        # -------------------------------------------------

        target_x += target_vx
        target_y += target_vy


        # -------------------------------------------------
        # BOUNCE TARGET OFF WALLS
        # -------------------------------------------------

        if target_x <= WORLD_MIN:

            target_x = WORLD_MIN

            target_vx *= -1


        elif target_x >= WORLD_MAX:

            target_x = WORLD_MAX

            target_vx *= -1


        if target_y <= WORLD_MIN:

            target_y = WORLD_MIN

            target_vy *= -1


        elif target_y >= WORLD_MAX:

            target_y = WORLD_MAX

            target_vy *= -1


    # -----------------------------------------------------
    # FINAL DISTANCE
    # -----------------------------------------------------

    final_distance = distance(
        fly_x,
        fly_y,
        target_x,
        target_y
    )


    if final_distance < TARGET_THRESHOLD:

        reached = True

        if steps_used == MAX_STEPS:

            steps_used = MAX_STEPS


    reduction = (
        (
            initial_distance -
            final_distance
        )
        /
        initial_distance
    ) * 100.0


    results.append(
        {
            "trial": trial,
            "initial": initial_distance,
            "final": final_distance,
            "reduction": reduction,
            "steps": steps_used,
            "success": reached
        }
    )


    status = (
        "SUCCESS"
        if reached
        else
        "FAILED"
    )


    print(
        f"Trial {trial:02d} | "
        f"{status:7s} | "
        f"initial={initial_distance:6.2f} | "
        f"final={final_distance:6.2f} | "
        f"reduction={reduction:6.2f}% | "
        f"steps={steps_used:2d}"
    )


# ---------------------------------------------------------
# STATISTICS
# ---------------------------------------------------------

successes = sum(
    r["success"]
    for r in results
)

success_rate = (
    successes /
    TRIALS
) * 100.0


initial_distances = np.array(
    [
        r["initial"]
        for r in results
    ]
)

final_distances = np.array(
    [
        r["final"]
        for r in results
    ]
)

reductions = np.array(
    [
        r["reduction"]
        for r in results
    ]
)


# ---------------------------------------------------------
# FINAL REPORT
# ---------------------------------------------------------

print()
print("=" * 70)
print("MOVING-TARGET BENCHMARK RESULTS")
print("=" * 70)

print()

print(
    f"Successful trials : "
    f"{successes}/{TRIALS}"
)

print(
    f"Success rate      : "
    f"{success_rate:.2f}%"
)

print(
    f"Mean reduction    : "
    f"{reductions.mean():.2f}%"
)

print(
    f"Median reduction  : "
    f"{np.median(reductions):.2f}%"
)

print(
    f"Mean final dist   : "
    f"{final_distances.mean():.2f}"
)

print(
    f"Median final dist : "
    f"{np.median(final_distances):.2f}"
)

print(
    f"Best final dist   : "
    f"{final_distances.min():.2f}"
)

print(
    f"Worst final dist  : "
    f"{final_distances.max():.2f}"
)

print()

print("=" * 70)

if success_rate == 100:

    print(
        "STATUS: ALL MOVING-TARGET TRIALS SUCCESSFUL"
    )

elif success_rate >= 80:

    print(
        "STATUS: STRONG MOVING-TARGET PERFORMANCE"
    )

elif success_rate >= 50:

    print(
        "STATUS: PARTIAL MOVING-TARGET PERFORMANCE"
    )

else:

    print(
        "STATUS: MOVING-TARGET CONTROLLER NEEDS IMPROVEMENT"
    )

print("=" * 70)