import numpy as np
from flybrain import FlyBrain

SIGMA = 0.08
BRAIN_STEPS = 50

TRIALS_PER_SPEED = 20
MAX_STEPS = 60

TARGET_THRESHOLD = 2.0

SEED = 42

WORLD_MIN = 5.0
WORLD_MAX = 95.0

TARGET_SPEEDS = [
    0.5,
    1.0,
    1.5,
    2.0,
    3.0
]


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
print("FLY-001 2D TARGET-SPEED STRESS TEST")
print("=" * 70)

print()
print(
    f"Trials per speed : {TRIALS_PER_SPEED}"
)

print(
    f"Maximum steps    : {MAX_STEPS}"
)

print(
    f"Target threshold : {TARGET_THRESHOLD}"
)

print(
    f"Speeds tested    : {TARGET_SPEEDS}"
)

print(
    f"Total trials     : "
    f"{TRIALS_PER_SPEED * len(TARGET_SPEEDS)}"
)

print(
    f"Random seed      : {SEED}"
)

print()


# ---------------------------------------------------------
# LOAD BRAIN
# ---------------------------------------------------------

brain = FlyBrain(
    device="auto"
)


# ---------------------------------------------------------
# LOAD DECODER
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
# RANDOM GENERATOR
# ---------------------------------------------------------

rng = np.random.default_rng(
    SEED
)


all_results = []


# ---------------------------------------------------------
# TEST EACH SPEED
# ---------------------------------------------------------

for target_speed in TARGET_SPEEDS:

    speed_results = []

    print()
    print("=" * 70)
    print(
        f"TARGET SPEED = {target_speed:.1f}"
    )
    print("=" * 70)


    for trial in range(
        1,
        TRIALS_PER_SPEED + 1
    ):

        # -------------------------------------------------
        # RANDOM WORLD
        # -------------------------------------------------

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


        # -------------------------------------------------
        # RANDOM TARGET DIRECTION
        # -------------------------------------------------

        angle = rng.uniform(
            0,
            2.0 * np.pi
        )

        target_vx = (
            np.cos(angle) *
            target_speed
        )

        target_vy = (
            np.sin(angle) *
            target_speed
        )


        reached = False
        steps_used = MAX_STEPS


        # -------------------------------------------------
        # NAVIGATION LOOP
        # -------------------------------------------------

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


            # Target reached
            if current_distance < TARGET_THRESHOLD:

                reached = True
                steps_used = step - 1

                break


            # -------------------------------------------------
            # WORLD → SENSORY INPUT
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
            # CREATE VISUAL INPUT
            # -------------------------------------------------

            stimulus = make_2d_input(
                brain.azimuth,
                sensory_x,
                sensory_y
            )


            # -------------------------------------------------
            # RUN BRAIN
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
            # DECODE X/Y
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
            # SAME CONTROLLER AS PREVIOUS BENCHMARK
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
            # BOUNCE TARGET
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


        # -------------------------------------------------
        # FINAL METRICS
        # -------------------------------------------------

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


        result = {
            "speed": target_speed,
            "trial": trial,
            "initial": initial_distance,
            "final": final_distance,
            "reduction": reduction,
            "steps": steps_used,
            "success": reached
        }


        speed_results.append(
            result
        )

        all_results.append(
            result
        )


        status = (
            "SUCCESS"
            if reached
            else "FAILED"
        )


        print(
            f"Trial {trial:02d} | "
            f"{status:7s} | "
            f"initial={initial_distance:6.2f} | "
            f"final={final_distance:6.2f} | "
            f"reduction={reduction:6.2f}% | "
            f"steps={steps_used:2d}"
        )


    # -----------------------------------------------------
    # SPEED SUMMARY
    # -----------------------------------------------------

    successes = sum(
        r["success"]
        for r in speed_results
    )

    success_rate = (
        successes /
        TRIALS_PER_SPEED
    ) * 100.0


    reductions = np.array(
        [
            r["reduction"]
            for r in speed_results
        ]
    )

    final_distances = np.array(
        [
            r["final"]
            for r in speed_results
        ]
    )


    print()
    print(
        f"Speed {target_speed:.1f} summary:"
    )

    print(
        f"  Success rate     : "
        f"{success_rate:.2f}%"
    )

    print(
        f"  Mean reduction   : "
        f"{reductions.mean():.2f}%"
    )

    print(
        f"  Mean final dist  : "
        f"{final_distances.mean():.2f}"
    )


# ---------------------------------------------------------
# OVERALL SUMMARY
# ---------------------------------------------------------

print()
print("=" * 70)
print("OVERALL SPEED-STRESS RESULTS")
print("=" * 70)

print()

for target_speed in TARGET_SPEEDS:

    speed_results = [
        r
        for r in all_results
        if r["speed"] == target_speed
    ]

    successes = sum(
        r["success"]
        for r in speed_results
    )

    success_rate = (
        successes /
        TRIALS_PER_SPEED
    ) * 100.0

    mean_reduction = np.mean(
        [
            r["reduction"]
            for r in speed_results
        ]
    )

    mean_final = np.mean(
        [
            r["final"]
            for r in speed_results
        ]
    )

    print(
        f"Speed {target_speed:.1f} | "
        f"{successes:02d}/{TRIALS_PER_SPEED} success | "
        f"success={success_rate:6.2f}% | "
        f"reduction={mean_reduction:6.2f}% | "
        f"final={mean_final:6.2f}"
    )


# ---------------------------------------------------------
# TOTAL
# ---------------------------------------------------------

total_successes = sum(
    r["success"]
    for r in all_results
)

total_trials = len(
    all_results
)

total_rate = (
    total_successes /
    total_trials
) * 100.0


print()
print("=" * 70)

print(
    f"TOTAL: "
    f"{total_successes}/{total_trials} successful"
)

print(
    f"OVERALL SUCCESS RATE: "
    f"{total_rate:.2f}%"
)

print("=" * 70)