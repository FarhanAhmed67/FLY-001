import numpy as np
from flybrain import FlyBrain
from fly_core import Fly001
from fly_vision import FlyVision


fly = Fly001()
vision = FlyVision(fly)
brain = fly.brain

target_receptors = 500

trials = 20
stage_steps = 50


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


left_stimulus = make_stimulus(-0.75)
right_stimulus = make_stimulus(0.75)

dark = np.zeros(
    vision.n,
    dtype=np.float32
)


def record_state(visual):

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

    return counts / stage_steps


results = {
    "LEFT": {
        "STIMULATED": [],
        "OFF_1S": [],
        "OFF_5S": []
    },

    "RIGHT": {
        "STIMULATED": [],
        "OFF_1S": [],
        "OFF_5S": []
    }
}


print(
    "FLY-001 direction decoder experiment"
)

print(
    "Trials:",
    trials
)

print(
    "Measurement:",
    stage_steps,
    "steps = 1000 ms"
)


for direction, stimulus in [
    ("LEFT", left_stimulus),
    ("RIGHT", right_stimulus)
]:

    print(
        f"\n========== {direction} =========="
    )

    for trial in range(trials):

        print(
            f"{direction}: "
            f"trial {trial + 1}/{trials}"
        )

        fly.reset()

        # STIMULATED
        stimulated = record_state(
            stimulus
        )

        # OFF 1 SECOND
        off_1s = record_state(
            dark
        )

        # Wait another 4 seconds
        for _ in range(200):
            fly.step(dark)

        # OFF 5 SECONDS
        off_5s = record_state(
            dark
        )

        results[
            direction
        ]["STIMULATED"].append(
            stimulated
        )

        results[
            direction
        ]["OFF_1S"].append(
            off_1s
        )

        results[
            direction
        ]["OFF_5S"].append(
            off_5s
        )


print(
    "\n\n========== FINDING DIRECTION-SENSITIVE NEURONS =========="
)


left = np.mean(
    results["LEFT"]["STIMULATED"],
    axis=0
)

right = np.mean(
    results["RIGHT"]["STIMULATED"],
    axis=0
)

difference = np.abs(
    left - right
)

ranking = np.argsort(
    difference
)[::-1]


print(
    "Largest LEFT/RIGHT differences:"
)

print(
    "\nRank | Neuron | LEFT | RIGHT | Difference"
)

for rank, neuron in enumerate(
    ranking[:50],
    start=1
):

    print(
        f"{rank:4d} | "
        f"{neuron:6d} | "
        f"{left[neuron]:6.2f} | "
        f"{right[neuron]:6.2f} | "
        f"{difference[neuron]:8.2f}"
    )


for top_n in [
    10,
    50,
    100,
    500,
    1000,
    5000
]:

    selected = ranking[:top_n]

    print(
        f"\n========== TOP {top_n} NEURONS =========="
    )

    for stage in [
        "STIMULATED",
        "OFF_1S",
        "OFF_5S"
    ]:

        left_stage = np.mean(
            results["LEFT"][stage],
            axis=0
        )

        right_stage = np.mean(
            results["RIGHT"][stage],
            axis=0
        )

        left_value = np.mean(
            left_stage[selected]
        )

        right_value = np.mean(
            right_stage[selected]
        )

        diff = abs(
            left_value - right_value
        )

        print(
            f"{stage:12} "
            f"LEFT={left_value:.3f} "
            f"RIGHT={right_value:.3f} "
            f"DIFF={diff:.3f}"
        )


print(
    "\nExperiment complete."
)