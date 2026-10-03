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

TARGET_SPEED = 3.0

PREDICTION_HORIZONS = [
    0.0,
    1.0,
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

    return signal.astype(np.float32)


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


def predict_direction(
    fly_x,
    fly_y,
    target_x,
    target_y
):

    dx = target_x - fly_x
    dy = target_y - fly_y

    return (
        np.clip(dx / 40.0, -1.0, 1.0),
        np.clip(dy / 40.0, -1.0, 1.0)
    )


def decode(
    activity,
    features,
    coef,
    intercept
):

    value = (
        activity[features] @ coef +
        intercept
    )

    return float(
        np.clip(
            value,
            -1.0,
            1.0
        )
    )


print("=" * 70)
print("FLY-001 PREDICTIVE PURSUIT TEST")
print("=" * 70)

print()
print(
    f"Target speed : {TARGET_SPEED}"
)

print(
    f"Trials/horizon : {TRIALS}"
)

print(
    f"Prediction horizons : "
    f"{PREDICTION_HORIZONS}"
)

print()


brain = FlyBrain(
    device="auto"
)


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


rng = np.random.default_rng(
    SEED
)


for horizon in PREDICTION_HORIZONS:

    results = []

    print()
    print("=" * 70)
    print(
        f"PREDICTION HORIZON = {horizon:.1f}"
    )
    print("=" * 70)


    for trial in range(
        1,
        TRIALS + 1
    ):

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


            if current_distance < TARGET_THRESHOLD:

                reached = True
                steps_used = step - 1

                break


            # -------------------------------------------------
            # PREDICT FUTURE TARGET POSITION
            # -------------------------------------------------

            predicted_x = (
                target_x +
                target_vx * horizon
            )

            predicted_y = (
                target_y +
                target_vy * horizon
            )


            predicted_x = np.clip(
                predicted_x,
                WORLD_MIN,
                WORLD_MAX
            )

            predicted_y = np.clip(
                predicted_y,
                WORLD_MIN,
                WORLD_MAX
            )


            sensory_x, sensory_y = predict_direction(
                fly_x,
                fly_y,
                predicted_x,
                predicted_y
            )


            # -------------------------------------------------
            # BRAIN
            # -------------------------------------------------

            brain.reset()


            stimulus = make_2d_input(
                brain.azimuth,
                sensory_x,
                sensory_y
            )


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
            # DECODE
            # -------------------------------------------------

            decoded_x = decode(
                activity,
                features_x,
                coef_x,
                intercept_x
            )

            decoded_y = decode(
                activity,
                features_y,
                coef_y,
                intercept_y
            )


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
            # CONTROLLER
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

            proposed_x = fly_x + move_x
            proposed_y = fly_y + move_y

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
            # BOUNCE
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


        final_distance = distance(
            fly_x,
            fly_y,
            target_x,
            target_y
        )


        if final_distance < TARGET_THRESHOLD:

            reached = True


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
                "success": reached,
                "final": final_distance,
                "reduction": reduction
            }
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
            f"reduction={reduction:6.2f}%"
        )


    successes = sum(
        r["success"]
        for r in results
    )


    success_rate = (
        successes /
        TRIALS
    ) * 100.0


    mean_final = np.mean(
        [
            r["final"]
            for r in results
        ]
    )


    mean_reduction = np.mean(
        [
            r["reduction"]
            for r in results
        ]
    )


    print()
    print(
        f"Horizon {horizon:.1f} summary:"
    )

    print(
        f"  Success rate   : "
        f"{success_rate:.2f}%"
    )

    print(
        f"  Mean reduction : "
        f"{mean_reduction:.2f}%"
    )

    print(
        f"  Mean final     : "
        f"{mean_final:.2f}"
    )


print()
print("=" * 70)
print("PREDICTIVE PURSUIT COMPLETE")
print("=" * 70)