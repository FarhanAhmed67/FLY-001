import numpy as np
from collections import Counter
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


def analyze_stage(visual):

    spike_total = 0

    unique_neurons = set()

    superclass_counts = Counter()

    motor_counts = {
        name: 0
        for name in brain.groups
    }

    for _ in range(stage_steps):

        fired = fly.step(
            visual
        )

        fired = np.asarray(
            fired,
            dtype=np.int64
        )

        spike_total += len(fired)

        unique_neurons.update(
            np.unique(fired).tolist()
        )

        superclass_counts.update(
            brain.superclass[fired]
        )

        for name, neuron_ids in brain.groups.items():

            for neuron_id in neuron_ids:

                motor_counts[name] += int(
                    np.sum(
                        fired == neuron_id
                    )
                )

    return {
        "spikes_per_step":
            spike_total / stage_steps,

        "unique_neurons":
            len(unique_neurons),

        "superclasses":
            Counter({
                key: value / stage_steps
                for key, value
                in superclass_counts.items()
            }),

        "motor":
            {
                key: value / stage_steps
                for key, value in motor_counts.items()
            }
    }


stages = [
    "INITIAL",
    "STIMULATED",
    "OFF_1S",
    "OFF_5S"
]


results = {
    direction: {
        stage: []
        for stage in stages
    }
    for direction in directions
}


print(
    "FLY-001 directional persistence experiment"
)

print(
    "500 receptors per stimulus"
)

print(
    "All measurement windows:",
    stage_steps,
    "steps =",
    stage_steps * 20,
    "ms"
)

print(
    "Trials:",
    trials
)


for direction_name in directions:

    print(
        f"\n========== {direction_name} =========="
    )

    stimulus = stimuli[direction_name]

    for trial in range(trials):

        print(
            f"{direction_name}: "
            f"trial {trial + 1}/{trials}"
        )

        fly.reset()

        # INITIAL
        results[direction_name]["INITIAL"].append(
            analyze_stage(dark)
        )

        # STIMULATED
        results[direction_name]["STIMULATED"].append(
            analyze_stage(stimulus)
        )

        # OFF 1 SECOND
        results[direction_name]["OFF_1S"].append(
            analyze_stage(dark)
        )

        # Wait another 4 seconds
        for _ in range(200):
            fly.step(dark)

        # OFF 5 SECONDS
        results[direction_name]["OFF_5S"].append(
            analyze_stage(dark)
        )


print(
    "\n\n========== STAGE SUMMARY =========="
)


for direction_name in directions:

    print(
        f"\n--- {direction_name} ---"
    )

    for stage in stages:

        stage_results = results[
            direction_name
        ][stage]

        spikes = np.mean([
            x["spikes_per_step"]
            for x in stage_results
        ])

        unique = np.mean([
            x["unique_neurons"]
            for x in stage_results
        ])

        print(
            f"{stage:12} "
            f"spikes/step = {spikes:8.1f} | "
            f"unique = {unique:8.1f}"
        )


print(
    "\n\n========== LEFT vs RIGHT DIFFERENCE =========="
)


for stage in stages:

    left_values = [
        x["spikes_per_step"]
        for x in results["LEFT"][stage]
    ]

    right_values = [
        x["spikes_per_step"]
        for x in results["RIGHT"][stage]
    ]

    left = np.mean(left_values)
    right = np.mean(right_values)

    difference = right - left

    percent = (
        abs(difference)
        / max((left + right) / 2, 1)
        * 100
    )

    print(
        f"\n{stage}"
    )

    print(
        f"  LEFT  = {left:.2f}"
    )

    print(
        f"  RIGHT = {right:.2f}"
    )

    print(
        f"  Difference = {difference:.2f}"
    )

    print(
        f"  Absolute difference = {percent:.2f}%"
    )


print(
    "\n\n========== MOTOR ACTIVITY =========="
)


for motor in brain.groups:

    print(
        f"\n{motor}"
    )

    for stage in stages:

        left = np.mean([
            x["motor"][motor]
            for x in results["LEFT"][stage]
        ])

        right = np.mean([
            x["motor"][motor]
            for x in results["RIGHT"][stage]
        ])

        print(
            f"  {stage:12} "
            f"LEFT={left:.3f} "
            f"RIGHT={right:.3f}"
        )


print(
    "\nExperiment complete."
)