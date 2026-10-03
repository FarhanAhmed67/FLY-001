import numpy as np
from flybrain import FlyBrain


POSITIONS = np.linspace(-1.0, 1.0, 21)
TRIALS = 5
STEPS = 50
TOP_N = 100


def generate_activity(brain, position):

    valid = np.where(np.isfinite(brain.azimuth))[0]

    distance = np.abs(
        brain.azimuth[valid] - position
    )

    selected = valid[
        np.argsort(distance)[:500]
    ]

    visual = np.zeros(
        len(brain.azimuth),
        dtype=np.float32
    )

    visual[selected] = 1.0

    brain.reset()

    neuron_counts = np.zeros(
        len(brain.positions),
        dtype=np.float32
    )

    for _ in range(STEPS):

        fired = brain.step(
            eye_drive=visual
        )

        if len(fired) > 0:
            neuron_counts += np.bincount(
                fired,
                minlength=len(brain.positions)
            )

    return neuron_counts


def main():

    print()
    print("=" * 70)
    print("FLY-001 NEURON-LEVEL SPATIAL MAP")
    print("=" * 70)

    brain = FlyBrain(device="auto")

    n_neurons = len(brain.positions)

    print()
    print(f"Neurons   : {n_neurons}")
    print(f"Receptors : {len(brain.visual)}")
    print(f"Positions : {len(POSITIONS)}")
    print(f"Trials    : {TRIALS}")
    print(f"Steps     : {STEPS}")

    all_activity = np.zeros(
        (len(POSITIONS), n_neurons),
        dtype=np.float32
    )

    for p_index, position in enumerate(POSITIONS):

        print()
        print(
            f"Position {p_index + 1}/"
            f"{len(POSITIONS)}: {position:+.2f}"
        )

        trials = []

        for trial in range(TRIALS):

            activity = generate_activity(
                brain,
                position
            )

            trials.append(activity)

            print(
                f"  Trial {trial + 1}: "
                f"{int(activity.sum())} spikes"
            )

        average_activity = np.mean(
            trials,
            axis=0
        )

        all_activity[p_index] = average_activity

        print(
            f"  Average: "
            f"{int(average_activity.sum())} spikes"
        )

    print()
    print("=" * 70)
    print("CALCULATING POSITION SENSITIVITY")
    print("=" * 70)

    # Each row = visual position
    # Each column = neuron

    neuron_means = all_activity.mean(axis=0)

    # Position sensitivity:
    # how much the neuron's activity varies
    # across different visual positions.

    neuron_std = all_activity.std(axis=0)

    # Select neurons with the highest spatial variation.

    top_indices = np.argsort(
        neuron_std
    )[::-1][:TOP_N]

    print()
    print(f"Top {TOP_N} position-sensitive neurons:")
    print()

    print(
        f"{'Rank':<6}"
        f"{'Neuron':<10}"
        f"{'Best position':<16}"
        f"{'Activity':<14}"
        f"{'Variation':<14}"
    )

    print("-" * 70)

    results = []

    for rank, neuron_id in enumerate(
        top_indices,
        start=1
    ):

        responses = all_activity[:, neuron_id]

        best_index = np.argmax(
            responses
        )

        best_position = POSITIONS[
            best_index
        ]

        best_activity = responses[
            best_index
        ]

        variation = neuron_std[
            neuron_id
        ]

        results.append([
            neuron_id,
            best_position,
            best_activity,
            variation
        ])

        print(
            f"{rank:<6}"
            f"{neuron_id:<10}"
            f"{best_position:+.2f}"
            f"{'':<12}"
            f"{best_activity:<14.1f}"
            f"{variation:<14.1f}"
        )

    print()
    print("=" * 70)
    print("POSITION TUNING SUMMARY")
    print("=" * 70)

    # Count how many top neurons prefer each position.

    preference_counts = {
        float(position): 0
        for position in POSITIONS
    }

    for neuron_id in top_indices:

        responses = all_activity[:, neuron_id]

        best_index = np.argmax(
            responses
        )

        best_position = float(
            POSITIONS[best_index]
        )

        preference_counts[
            best_position
        ] += 1

    for position in POSITIONS:

        print(
            f"{position:+.2f}"
            f" -> "
            f"{preference_counts[float(position)]}"
            f" top neurons"
        )

    print()
    print("=" * 70)
    print("GLOBAL STATISTICS")
    print("=" * 70)

    print(
        f"Mean neuron variation : "
        f"{neuron_std.mean():.3f}"
    )

    print(
        f"Maximum neuron variation: "
        f"{neuron_std.max():.3f}"
    )

    print(
        f"Most selective neuron : "
        f"{top_indices[0]}"
    )

    print(
        f"Preferred position     : "
        f"{results[0][1]:+.2f}"
    )

    print(
        f"Peak activity          : "
        f"{results[0][2]:.1f}"
    )

    print(
        f"Activity variation     : "
        f"{results[0][3]:.1f}"
    )

    # Save complete activity matrix.

    np.savez(
        "fly_neuron_spatial_map.npz",
        positions=POSITIONS,
        activity=all_activity,
        neuron_std=neuron_std,
        top_indices=top_indices
    )

    # Save human-readable top neuron table.

    with open(
        "fly_top_spatial_neurons.txt",
        "w"
    ) as f:

        f.write(
            "FLY-001 TOP POSITION-SENSITIVE NEURONS\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            "Rank,Neuron,BestPosition,PeakActivity,Variation\n"
        )

        for rank, row in enumerate(
            results,
            start=1
        ):

            neuron_id = row[0]
            best_position = row[1]
            peak_activity = row[2]
            variation = row[3]

            f.write(
                f"{rank},{neuron_id},"
                f"{best_position:+.2f},"
                f"{peak_activity:.2f},"
                f"{variation:.2f}\n"
            )

    print()
    print("Saved:")
    print("  fly_neuron_spatial_map.npz")
    print("  fly_top_spatial_neurons.txt")

    print()
    print("=" * 70)
    print("NEURON SPATIAL MAP COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()