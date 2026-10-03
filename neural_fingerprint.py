import numpy as np
from fly_core import Fly001
from fly_vision import FlyVision


fly = Fly001()
vision = FlyVision(fly)

brain = fly.brain

target_receptors = 500

trials = 10
stage_steps = 50

directions = {
    "LEFT": -0.75,
    "RIGHT": 0.75
}


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

    visual[selected] = 1.0

    return visual


dark = np.zeros(
    vision.n,
    dtype=np.float32
)


stimuli = {
    name: make_stimulus(direction)
    for name, direction in directions.items()
}


def get_state(visual):

    counts = np.zeros(
        len(brain.cell_type),
        dtype=np.float32
    )

    for _ in range(stage_steps):

        fired = fly.step(
            visual
        )

        fired = np.asarray(
            fired,
            dtype=np.int64
        )

        if len(fired) > 0:

            unique, frequency = np.unique(
                fired,
                return_counts=True
            )

            counts[unique] += frequency

    return counts


def jaccard(a, b):

    active_a = a > 0
    active_b = b > 0

    intersection = np.sum(
        active_a & active_b
    )

    union = np.sum(
        active_a | active_b
    )

    if union == 0:
        return 1.0

    return intersection / union


def cosine_similarity(a, b):

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return np.dot(a, b) / denominator


states = {
    direction: {
        stage: []
        for stage in [
            "INITIAL",
            "STIMULATED",
            "OFF_1S",
            "OFF_5S"
        ]
    }
    for direction in directions
}


print(
    "FLY-001 neural fingerprint experiment"
)

print(
    "500 receptors per stimulus"
)

print(
    "Measurement window:",
    stage_steps,
    "steps"
)

print(
    "Trials:",
    trials
)


for direction_name in directions:

    print(
        f"\n========== {direction_name} =========="
    )

    stimulus = stimuli[
        direction_name
    ]

    for trial in range(trials):

        print(
            f"{direction_name}: "
            f"trial {trial + 1}/{trials}"
        )

        fly.reset()

        # INITIAL
        states[
            direction_name
        ]["INITIAL"].append(
            get_state(dark)
        )

        # STIMULATED
        states[
            direction_name
        ]["STIMULATED"].append(
            get_state(stimulus)
        )

        # OFF 1 SECOND
        states[
            direction_name
        ]["OFF_1S"].append(
            get_state(dark)
        )

        # Wait another 4 seconds
        for _ in range(200):
            fly.step(dark)

        # OFF 5 SECONDS
        states[
            direction_name
        ]["OFF_5S"].append(
            get_state(dark)
        )


print(
    "\n\n========== WITHIN-DIRECTION PERSISTENCE =========="
)

for direction in directions:

    print(
        f"\n--- {direction} ---"
    )

    for off_stage in [
        "OFF_1S",
        "OFF_5S"
    ]:

        jaccard_values = []
        cosine_values = []

        for trial in range(trials):

            stimulated = states[
                direction
            ]["STIMULATED"][trial]

            off = states[
                direction
            ][off_stage][trial]

            jaccard_values.append(
                jaccard(
                    stimulated,
                    off
                )
            )

            cosine_values.append(
                cosine_similarity(
                    stimulated,
                    off
                )
            )

        print(
            f"{off_stage}:"
        )

        print(
            f"  Jaccard = "
            f"{np.mean(jaccard_values):.4f}"
        )

        print(
            f"  Cosine  = "
            f"{np.mean(cosine_values):.4f}"
        )


print(
    "\n\n========== LEFT vs RIGHT =========="
)


for stage in [
    "INITIAL",
    "STIMULATED",
    "OFF_1S",
    "OFF_5S"
]:

    jaccard_values = []
    cosine_values = []

    for trial in range(trials):

        left = states[
            "LEFT"
        ][stage][trial]

        right = states[
            "RIGHT"
        ][stage][trial]

        jaccard_values.append(
            jaccard(
                left,
                right
            )
        )

        cosine_values.append(
            cosine_similarity(
                left,
                right
            )
        )

    print(
        f"\n{stage}"
    )

    print(
        f"  Jaccard = "
        f"{np.mean(jaccard_values):.4f}"
    )

    print(
        f"  Cosine  = "
        f"{np.mean(cosine_values):.4f}"
    )


print(
    "\nExperiment complete."
)