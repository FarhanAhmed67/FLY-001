import numpy as np
from fly_core import Fly001
from fly_vision import FlyVision


fly = Fly001()
vision = FlyVision(fly)

directions = np.linspace(-1.0, 1.0, 9)

target_receptors = 500
intensity = 1.0

trials = 20
steps_per_position = 5


def make_stimulus(direction):

    distance = np.abs(
        vision.azimuth - direction
    )

    order = np.argsort(distance)

    selected = order[:target_receptors]

    visual = np.zeros(
        vision.n,
        dtype=np.float32
    )

    visual[selected] = intensity

    return visual


def run_trial(sequence):

    fly.reset()

    spike_counts = []

    for visual in sequence:

        total_spikes = 0

        for _ in range(steps_per_position):

            fired = fly.step(visual)

            total_spikes += len(fired)

        spike_counts.append(total_spikes)

    return spike_counts


print("FLY-001 temporal response experiment")
print("Target receptors:", target_receptors)
print("Trials:", trials)
print("Steps per position:", steps_per_position)


stimuli = {
    direction: make_stimulus(direction)
    for direction in directions
}


# --------------------------------------------------
# STATIONARY STIMULUS
# --------------------------------------------------

print("\n========== STATIONARY ==========")

stationary_results = []

for direction in directions:

    sequence = [
        stimuli[direction]
        for _ in directions
    ]

    trial_results = []

    for _ in range(trials):

        result = run_trial(sequence)

        trial_results.append(result)

    average = np.mean(
        trial_results,
        axis=0
    )

    stationary_results.append(average)

    print(
        f"{direction:+.2f}:",
        " ".join(
            f"{x:.0f}"
            for x in average
        )
    )


# --------------------------------------------------
# MOVING STIMULUS
# --------------------------------------------------

print("\n========== MOVING ==========")

moving_results = []

sequence = [
    stimuli[direction]
    for direction in directions
]

for _ in range(trials):

    result = run_trial(sequence)

    moving_results.append(result)

moving_average = np.mean(
    moving_results,
    axis=0
)

print(
    "Moving:",
    " ".join(
        f"{x:.0f}"
        for x in moving_average
    )
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n========== SUMMARY ==========")

print("Positions:")

print(
    " ".join(
        f"{direction:+.2f}"
        for direction in directions
    )
)

print("\nMoving average spikes:")

print(
    " ".join(
        f"{x:.0f}"
        for x in moving_average
    )
)

print("\nStationary average total spikes:")

for direction, result in zip(
    directions,
    stationary_results
):

    print(
        f"{direction:+.2f}: "
        f"{np.mean(result):.1f}"
    )