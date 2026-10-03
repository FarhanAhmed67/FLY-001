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

on_steps = 50
off_1s_steps = 50
off_remaining_steps = 200


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


def analyze_spikes(fired):

    fired = np.asarray(
        fired,
        dtype=np.int64
    )

    unique = np.unique(fired)

    superclass_counts = Counter(
        brain.superclass[fired]
    )

    motor_counts = {}

    for name, neuron_ids in brain.groups.items():

        count = 0

        for neuron_id in neuron_ids:
            count += np.sum(
                fired == neuron_id
            )

        motor_counts[name] = int(count)

    return {
        "spikes": len(fired),
        "unique_neurons": len(unique),
        "superclasses": superclass_counts,
        "motor": motor_counts
    }


def run_stage(visual, steps):

    spike_total = 0
    unique_neurons = set()

    superclass_counts = Counter()
    motor_counts = {
        name: 0
        for name in brain.groups
    }

    for _ in range(steps):

        fired = fly.step(
            visual
        )

        result = analyze_spikes(
            fired
        )

        spike_total += result["spikes"]

        unique_neurons.update(
            np.unique(fired).tolist()
        )

        superclass_counts.update(
            result["superclasses"]
        )

        for name in motor_counts:
            motor_counts[name] += (
                result["motor"][name]
            )

    return {
        "total_spikes": spike_total,
        "unique_neurons": len(unique_neurons),
        "superclasses": superclass_counts,
        "motor": motor_counts
    }


print("FLY-001 persistence mapping experiment")
print("Direction:", direction)
print("Active receptors:", target_receptors)
print("Trials:", trials)

print("\nBrain neurons:", brain.n)


stage_names = [
    "INITIAL",
    "STIMULATED",
    "OFF_1S",
    "OFF_5S"
]


all_stage_data = {
    stage: []
    for stage in stage_names
}


for trial in range(trials):

    print(
        f"\nRunning trial {trial + 1}/{trials}..."
    )

    fly.reset()

    # -----------------------------------------
    # INITIAL
    # -----------------------------------------

    initial = run_stage(
        dark,
        1
    )

    all_stage_data["INITIAL"].append(
        initial
    )

    # -----------------------------------------
    # STIMULATED
    # 1000 ms
    # -----------------------------------------

    stimulated = run_stage(
        light,
        on_steps
    )

    all_stage_data["STIMULATED"].append(
        stimulated
    )

    # -----------------------------------------
    # OFF FOR 1 SECOND
    # -----------------------------------------

    off_1s = run_stage(
        dark,
        off_1s_steps
    )

    all_stage_data["OFF_1S"].append(
        off_1s
    )

    # -----------------------------------------
    # OFF FOR ANOTHER 4 SECONDS
    # -----------------------------------------

    off_5s = run_stage(
        dark,
        off_remaining_steps
    )

    all_stage_data["OFF_5S"].append(
        off_5s
    )


# ==================================================
# SUMMARY
# ==================================================

print("\n\n========== STAGE SUMMARY ==========")

for stage in stage_names:

    results = all_stage_data[stage]

    avg_spikes = np.mean([
        x["total_spikes"]
        for x in results
    ])

    avg_unique = np.mean([
        x["unique_neurons"]
        for x in results
    ])

    print(
        f"\n{stage}"
    )

    print(
        f"Average total spikes: "
        f"{avg_spikes:.1f}"
    )

    print(
        f"Average unique neurons: "
        f"{avg_unique:.1f}"
    )


# ==================================================
# SUPERCLASS ANALYSIS
# ==================================================

print("\n\n========== SUPERCLASS ACTIVITY ==========")

all_superclasses = set()

for stage in stage_names:

    for result in all_stage_data[stage]:

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

    for stage in stage_names:

        values = []

        for result in all_stage_data[stage]:

            values.append(
                result["superclasses"].get(
                    superclass,
                    0
                )
            )

        print(
            f"  {stage:10} "
            f"{np.mean(values):.1f}"
        )


# ==================================================
# MOTOR ACTIVITY
# ==================================================

print("\n\n========== MOTOR ACTIVITY ==========")

for name in brain.groups:

    print(
        f"\n{name}"
    )

    for stage in stage_names:

        values = []

        for result in all_stage_data[stage]:

            values.append(
                result["motor"][name]
            )

        print(
            f"  {stage:10} "
            f"{np.mean(values):.2f}"
        )


print("\n\nExperiment complete.")