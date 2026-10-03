import math
import numpy as np
from flybrain import FlyBrain
from sklearn.linear_model import Ridge


# ============================================================
# FLY-001 PREDICTION HORIZON SWEEP
# ============================================================

SIGMA = 0.08
BRAIN_STEPS = 50
TRIALS = 20
MAX_STEPS = 60

TARGET_SPEED = 3.0
TARGET_THRESHOLD = 2.0

WORLD_MIN = 5.0
WORLD_MAX = 95.0

SEED = 42

HORIZONS = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
]


# ============================================================
# SOFT VISUAL ENCODING
# ============================================================

def soft_signal(azimuth, position):
    azimuth = np.asarray(azimuth, dtype=np.float32)

    distance = np.abs(azimuth - position)

    stimulus = np.exp(
        -(distance ** 2) / (2 * SIGMA ** 2)
    )

    stimulus[~np.isfinite(azimuth)] = 0.0

    maximum = stimulus.max()

    if maximum > 0:
        stimulus /= maximum

    return stimulus.astype(np.float32)


def make_2d_input(azimuth, x, y):
    """
    Project-level 2D representation.

    First half  = horizontal signal
    Second half = vertical signal
    """

    n = len(azimuth)
    stimulus = np.zeros(n, dtype=np.float32)

    half = n // 2

    horizontal = soft_signal(azimuth, x)
    vertical = soft_signal(azimuth, y)

    stimulus[:half] = horizontal[:half]
    stimulus[half:] = vertical[half:]

    return stimulus


# ============================================================
# LOAD TRAINED DECODER
# ============================================================

print("=" * 70)
print("FLY-001 PREDICTION HORIZON SWEEP")
print("=" * 70)

print()
print("Target speed        :", TARGET_SPEED)
print("Trials/horizon      :", TRIALS)
print("Prediction horizons :", HORIZONS)
print()


data = np.load(
    "fly_2d_continuous_decoder.npz"
)

features_x = data["features_x"]
features_y = data["features_y"]

coef_x = data["coef_x"]
coef_y = data["coef_y"]

intercept_x = float(data["intercept_x"])
intercept_y = float(data["intercept_y"])


def decode(X):
    x = X[:, features_x] @ coef_x + intercept_x
    y = X[:, features_y] @ coef_y + intercept_y

    return float(x[0]), float(y[0])


# ============================================================
# BRAIN
# ============================================================

brain = FlyBrain(device="auto")

azimuth = np.asarray(
    brain.azimuth,
    dtype=np.float32
)

N_VISUAL = len(brain.visual)


# ============================================================
# RANDOM TARGET
# ============================================================

rng = np.random.default_rng(SEED)


def random_target_velocity():
    angle = rng.uniform(
        0,
        2 * math.pi
    )

    return (
        TARGET_SPEED * math.cos(angle),
        TARGET_SPEED * math.sin(angle)
    )


# ============================================================
# RUN ONE TRIAL
# ============================================================

def run_trial(horizon):

    fly_x = rng.uniform(
        WORLD_MIN,
        WORLD_MAX
    )

    fly_y = rng.uniform(
        WORLD_MIN,
        WORLD_MAX
    )

    target_x = rng.uniform(
        WORLD_MIN,
        WORLD_MAX
    )

    target_y = rng.uniform(
        WORLD_MIN,
        WORLD_MAX
    )

    target_vx, target_vy = random_target_velocity()

    initial_distance = math.hypot(
        target_x - fly_x,
        target_y - fly_y
    )

    final_distance = initial_distance

    brain.reset()

    for step in range(1, MAX_STEPS + 1):

        # ----------------------------------------------------
        # PREDICT TARGET POSITION
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # TARGET RELATIVE TO FLY
        # ----------------------------------------------------

        dx = predicted_x - fly_x
        dy = predicted_y - fly_y

        sensory_x = np.clip(
            dx / 40.0,
            -1.0,
            1.0
        )

        sensory_y = np.clip(
            dy / 40.0,
            -1.0,
            1.0
        )

        visual_input = make_2d_input(
            azimuth,
            sensory_x,
            sensory_y
        )

        # ----------------------------------------------------
        # RUN FLY BRAIN
        # ----------------------------------------------------

        recordings = []

        for _ in range(BRAIN_STEPS):

            fired = brain.step(
                eye_drive=visual_input
            )

            recordings.append(
                fired
            )

        # ----------------------------------------------------
        # BUILD POPULATION VECTOR
        # ----------------------------------------------------

        population = np.zeros(
            166700,
            dtype=np.float32
        )

        for fired in recordings:

            for neuron_id in fired:

                population[int(neuron_id)] += 1.0

        population = population.reshape(
            1,
            -1
        )

        # ----------------------------------------------------
        # DECODE POSITION
        # ----------------------------------------------------

        decoded_x, decoded_y = decode(
            population
        )

        decoded_x = float(
            np.clip(decoded_x, -1.0, 1.0)
        )

        decoded_y = float(
            np.clip(decoded_y, -1.0, 1.0)
        )

        # ----------------------------------------------------
        # NORMALIZE DIRECTION
        # ----------------------------------------------------

        magnitude = math.hypot(
            decoded_x,
            decoded_y
        )

        if magnitude > 1e-6:

            direction_x = (
                decoded_x / magnitude
            )

            direction_y = (
                decoded_y / magnitude
            )

        else:

            direction_x = 0.0
            direction_y = 0.0

        # ----------------------------------------------------
        # ADAPTIVE MOVEMENT
        # ----------------------------------------------------

        current_distance = math.hypot(
            target_x - fly_x,
            target_y - fly_y
        )

        if current_distance > 12:
            scale = 4.0

        elif current_distance > 6:
            scale = 3.0

        elif current_distance > 3:
            scale = 2.0

        else:
            scale = 1.0

        move_x = direction_x * scale
        move_y = direction_y * scale

        # ----------------------------------------------------
        # OVERSHOOT PROTECTION
        # ----------------------------------------------------

        proposed_x = np.clip(
            fly_x + move_x,
            WORLD_MIN,
            WORLD_MAX
        )

        proposed_y = np.clip(
            fly_y + move_y,
            WORLD_MIN,
            WORLD_MAX
        )

        old_distance = math.hypot(
            target_x - fly_x,
            target_y - fly_y
        )

        new_distance = math.hypot(
            target_x - proposed_x,
            target_y - proposed_y
        )

        if new_distance > old_distance:

            move_x *= 0.5
            move_y *= 0.5

        # ----------------------------------------------------
        # MOVE FLY
        # ----------------------------------------------------

        fly_x = np.clip(
            fly_x + move_x,
            WORLD_MIN,
            WORLD_MAX
        )

        fly_y = np.clip(
            fly_y + move_y,
            WORLD_MIN,
            WORLD_MAX
        )

        # ----------------------------------------------------
        # MOVE TARGET
        # ----------------------------------------------------

        target_x += target_vx
        target_y += target_vy

        # ----------------------------------------------------
        # BOUNDARY BOUNCE
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # DISTANCE
        # ----------------------------------------------------

        final_distance = math.hypot(
            target_x - fly_x,
            target_y - fly_y
        )

        if final_distance <= TARGET_THRESHOLD:

            return (
                True,
                initial_distance,
                final_distance,
                step
            )

    return (
        False,
        initial_distance,
        final_distance,
        MAX_STEPS
    )


# ============================================================
# HORIZON SWEEP
# ============================================================

all_results = []


for horizon in HORIZONS:

    print()
    print("=" * 70)
    print(
        f"PREDICTION HORIZON = {horizon:.1f}"
    )
    print("=" * 70)

    successes = 0
    reductions = []
    final_distances = []

    for trial in range(
        1,
        TRIALS + 1
    ):

        success, initial, final, steps = run_trial(
            horizon
        )

        reduction = (
            (initial - final) /
            initial *
            100.0
        )

        reductions.append(
            reduction
        )

        final_distances.append(
            final
        )

        if success:
            successes += 1
            status = "SUCCESS"
        else:
            status = "FAILED"

        print(
            f"Trial {trial:02d} | "
            f"{status:7s} | "
            f"initial={initial:6.2f} | "
            f"final={final:6.2f} | "
            f"reduction={reduction:6.2f}% | "
            f"steps={steps:02d}"
        )

    success_rate = (
        successes /
        TRIALS *
        100.0
    )

    mean_reduction = np.mean(
        reductions
    )

    mean_final = np.mean(
        final_distances
    )

    median_final = np.median(
        final_distances
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

    print(
        f"  Median final   : "
        f"{median_final:.2f}"
    )

    all_results.append(
        (
            horizon,
            successes,
            success_rate,
            mean_reduction,
            mean_final,
            median_final
        )
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("HORIZON SWEEP COMPLETE")
print("=" * 70)

print()
print(
    "Horizon | Success | "
    "Mean Reduction | Mean Final | Median Final"
)
print("-" * 70)

for result in all_results:

    horizon, successes, rate, reduction, mean_final, median_final = result

    print(
        f"{horizon:7.1f} | "
        f"{rate:6.2f}% | "
        f"{reduction:13.2f}% | "
        f"{mean_final:10.2f} | "
        f"{median_final:12.2f}"
    )

best = max(
    all_results,
    key=lambda x: (
        x[2],
        -x[4]
    )
)

print()
print(
    f"Best tested horizon by success rate: "
    f"{best[0]:.1f}"
)

print(
    f"Success rate: {best[2]:.2f}%"
)

print(
    f"Mean final distance: {best[4]:.2f}"
)

print()
print("=" * 70)