import numpy as np
from collections import Counter
from fly_core import Fly001
from fly_vision import FlyVision


fly = Fly001()
vision = FlyVision(fly)

brain = fly.brain

direction = 0.0
target_receptors = 500

trials = 10

stage_steps = 50


def make_stimulus():

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


light = make_stimulus()

dark = np.zeros(
    vision.n,
    dtype=np.float32
)


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
                for key, value
                in motor_counts.items()
            }
    }


stages = {
    "INITIAL": dark,
    "STIMULATED": light,
    "OFF_1S": dark,
    "OFF_5S": dark
}


results = {
    stage: []
    for stage in stages
}


print(
    "FLY-001 normalized persistence experiment"
)

print(
    "All stages use:",
    stage_steps,
    "steps =",
    stage_steps * 20,
    "ms"
)

print(
    "Trials:",
    trials
)


for trial in range(trials):

    print(
        f"Running trial {trial + 1}/{trials}..."
    )

    fly.reset()

    # INITIAL
    results["INITIAL"].append(
        analyze_stage(dark)
    )

    # STIMULATED
    results["STIMULATED"].append(
        analyze_stage(light)
    )

    # OFF 1 SECOND
    results["OFF_1S"].append(
        analyze_stage(dark)
    )

    # OFF 5 SECONDS
    #
    # First advance four seconds,
    # then measure a fresh 1-second window.
    for _ in range(200):
        fly.step(dark)

    results["OFF_5S"].append(
        analyze_stage(dark)
    )


print("\n========== NORMALIZED STAGE SUMMARY ==========")


for stage in stages:

    stage_results = results[stage]

    spikes = np.mean([
        x["spikes_per_step"]
        for x in stage_results
    ])

    unique = np.mean([
        x["unique_neurons"]
        for x in stage_results
    ])

    print(
        f"\n{stage}"
    )

    print(
        f"Average spikes/step: "
        f"{spikes:.1f}"
    )

    print(
        f"Average unique neurons: "
        f"{unique:.1f}"
    )


print("\n========== SUPERCLASS SPIKES / STEP ==========")


all_superclasses = set()

for stage in stages:

    for result in results[stage]:

        all_superclasses.update(
            result["superclasses"].keys()
        )


for superclass in sorted(
    all_superclasses,
    key=str
):

    print(
        f"\n{superclass}"
    )

    for stage in stages:

        values = [
            result["superclasses"].get(
                superclass,
                0
            )
            for result in results[stage]
        ]

        print(
            f"  {stage:10} "
            f"{np.mean(values):.2f}"
        )


print("\n========== MOTOR SPIKES / STEP ==========")


for name in brain.groups:

    print(
        f"\n{name}"
    )

    for stage in stages:

        values = [
            result["motor"][name]
            for result in results[stage]
        ]

        print(
            f"  {stage:10} "
            f"{np.mean(values):.3f}"
        )


print("\nExperiment complete.")