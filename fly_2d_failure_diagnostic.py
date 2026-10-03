import numpy as np
from flybrain import FlyBrain

SIGMA = 0.08
BRAIN_STEPS = 50

TARGET_SPEED = 3.0

TRIALS = 20
MAX_STEPS = 60

TARGET_THRESHOLD = 2.0

SEED = 42

WORLD_MIN = 5.0
WORLD_MAX = 95.0


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


def world_to_target(
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


def predict(
    activity,
    features,
    coef,
    intercept
):

    return float(
        np.clip(
            activity[features] @ coef + intercept,
            -1.0,
            1.0
        )
    )


print("=" * 70)
print("FLY-001 FAILURE DIAGNOSTIC")
print("=" * 70)

print()
print(f"Target speed : {TARGET_SPEED}")
print(f"Trials       : {TRIALS}")
print(f"Max steps    : {MAX_STEPS}")
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


failed_count = 0


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


    trajectory = []

    reached = False


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
            break


        sensory_x, sensory_y = world_to_target(
            fly_x,
            fly_y,
            target_x,
            target_y
        )


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


        old_fly_x = fly_x
        old_fly_y = fly_y

        old_target_x = target_x
        old_target_y = target_y


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


        target_x += target_vx
        target_y += target_vy


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


        new_distance = distance(
            fly_x,
            fly_y,
            target_x,
            target_y
        )


        trajectory.append(
            {
                "step": step,
                "distance": current_distance,
                "new_distance": new_distance,
                "fly_x": old_fly_x,
                "fly_y": old_fly_y,
                "target_x": old_target_x,
                "target_y": old_target_y,
                "decoded_x": decoded_x,
                "decoded_y": decoded_y,
                "move_x": move_x,
                "move_y": move_y,
                "target_vx": target_vx,
                "target_vy": target_vy
            }
        )


    final_distance = distance(
        fly_x,
        fly_y,
        target_x,
        target_y
    )


    if not reached:

        failed_count += 1

        print()
        print("=" * 70)
        print(
            f"FAILED TRIAL {trial}"
        )
        print("=" * 70)

        print(
            f"Initial distance : "
            f"{initial_distance:.2f}"
        )

        print(
            f"Final distance   : "
            f"{final_distance:.2f}"
        )

        print()

        print(
            "STEP | DIST | DX | DY | "
            "MOVE X | MOVE Y | "
            "TARGET VX | TARGET VY"
        )

        print("-" * 90)


        # Show every 5th step
        # plus the final 5 steps

        shown = set()

        for item in trajectory:

            if (
                item["step"] % 5 == 0
                or item["step"] >= MAX_STEPS - 4
            ):

                shown.add(
                    item["step"]
                )

                print(
                    f"{item['step']:4d} | "
                    f"{item['distance']:5.2f} | "
                    f"{item['decoded_x']:+.3f} | "
                    f"{item['decoded_y']:+.3f} | "
                    f"{item['move_x']:+6.2f} | "
                    f"{item['move_y']:+6.2f} | "
                    f"{item['target_vx']:+7.2f} | "
                    f"{item['target_vy']:+7.2f}"
                )


        print()

        # Average movement and target velocity

        moves = np.array(
            [
                [
                    item["move_x"],
                    item["move_y"]
                ]
                for item in trajectory
            ]
        )


        target_velocities = np.array(
            [
                [
                    item["target_vx"],
                    item["target_vy"]
                ]
                for item in trajectory
            ]
        )


        avg_move = np.mean(
            np.linalg.norm(
                moves,
                axis=1
            )
        )


        avg_target_speed = np.mean(
            np.linalg.norm(
                target_velocities,
                axis=1
            )
        )


        print(
            f"Average fly movement/step : "
            f"{avg_move:.2f}"
        )

        print(
            f"Target speed              : "
            f"{avg_target_speed:.2f}"
        )

        print()


print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)

print(
    f"Failed trials: "
    f"{failed_count}/{TRIALS}"
)

print("=" * 70)