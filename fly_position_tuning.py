import numpy as np
import matplotlib.pyplot as plt

from flybrain import FlyBrain


POSITIONS = np.linspace(-1.0, 1.0, 21)

TRIALS = 10

STEPS = 50

TOP_N = 12


def create_visual(brain, position):

    valid = np.where(
        np.isfinite(brain.azimuth)
    )[0]

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

    return visual


def measure_activity(brain, visual):

    brain.reset()

    counts = np.zeros(
        len(brain.positions),
        dtype=np.float32
    )

    for _ in range(STEPS):

        fired = brain.step(
            eye_drive=visual
        )

        if len(fired) > 0:

            counts += np.bincount(
                fired,
                minlength=len(brain.positions)
            )

    return counts


def main():

    print()
    print("=" * 70)
    print("FLY-001 POSITION TUNING CURVES")
    print("=" * 70)

    brain = FlyBrain(device="auto")

    n_neurons = len(brain.positions)

    print()
    print(f"Neurons   : {n_neurons}")
    print(f"Positions : {len(POSITIONS)}")
    print(f"Trials    : {TRIALS}")
    print(f"Steps     : {STEPS}")

    # ---------------------------------------------------------
    # Load neurons discovered in the previous experiment.
    # ---------------------------------------------------------

    previous = np.load(
        "fly_neuron_spatial_map.npz"
    )

    candidate_neurons = previous[
        "top_indices"
    ]

    print()
    print(
        f"Loaded {len(candidate_neurons)} "
        "candidate neurons."
    )

    # ---------------------------------------------------------
    # Measure every candidate neuron at every position.
    # ---------------------------------------------------------

    activity = np.zeros(
        (
            len(POSITIONS),
            TRIALS,
            len(candidate_neurons)
        ),
        dtype=np.float32
    )

    for p_index, position in enumerate(POSITIONS):

        print()
        print(
            f"Position {p_index + 1}/"
            f"{len(POSITIONS)}: "
            f"{position:+.2f}"
        )

        visual = create_visual(
            brain,
            position
        )

        for trial in range(TRIALS):

            counts = measure_activity(
                brain,
                visual
            )

            activity[
                p_index,
                trial
            ] = counts[
                candidate_neurons
            ]

        print(
            f"  Mean total spikes: "
            f"{activity[p_index].sum(axis=1).mean():.0f}"
        )

    # ---------------------------------------------------------
    # Average across trials.
    # ---------------------------------------------------------

    mean_activity = activity.mean(
        axis=1
    )

    # ---------------------------------------------------------
    # Calculate tuning strength.
    #
    # A neuron is considered more selective when:
    #
    #   maximum response - minimum response
    #
    # is large relative to its overall response.
    # ---------------------------------------------------------

    tuning_strength = np.zeros(
        len(candidate_neurons)
    )

    preferred_positions = np.zeros(
        len(candidate_neurons)
    )

    for n in range(len(candidate_neurons)):

        response = mean_activity[:, n]

        peak = response.max()

        minimum = response.min()

        tuning_strength[n] = (
            peak - minimum
        ) / max(
            peak,
            1.0
        )

        preferred_positions[n] = POSITIONS[
            np.argmax(response)
        ]

    # ---------------------------------------------------------
    # Rank candidates by tuning strength.
    # ---------------------------------------------------------

    ranking = np.argsort(
        tuning_strength
    )[::-1]

    selected = ranking[:TOP_N]

    print()
    print("=" * 70)
    print("TOP TUNED NEURONS")
    print("=" * 70)

    print()

    print(
        f"{'Rank':<6}"
        f"{'Neuron':<10}"
        f"{'Preferred':<12}"
        f"{'Peak':<10}"
        f"{'Minimum':<10}"
        f"{'Selectivity':<12}"
    )

    print("-" * 70)

    for rank, index in enumerate(
        selected,
        start=1
    ):

        neuron_id = candidate_neurons[index]

        response = mean_activity[
            :,
            index
        ]

        peak = response.max()

        minimum = response.min()

        preferred = preferred_positions[
            index
        ]

        selectivity = tuning_strength[
            index
        ]

        print(
            f"{rank:<6}"
            f"{neuron_id:<10}"
            f"{preferred:+.2f}"
            f"{'':<8}"
            f"{peak:<10.1f}"
            f"{minimum:<10.1f}"
            f"{selectivity:<12.3f}"
        )

    # ---------------------------------------------------------
    # Test reproducibility.
    #
    # Split trials into two independent groups.
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("REPRODUCIBILITY TEST")
    print("=" * 70)

    correlations = []

    half = TRIALS // 2

    for index in selected:

        first_half = activity[
            :,
            :half,
            index
        ].mean(axis=1)

        second_half = activity[
            :,
            half:,
            index
        ].mean(axis=1)

        if (
            np.std(first_half) > 0
            and np.std(second_half) > 0
        ):

            correlation = np.corrcoef(
                first_half,
                second_half
            )[0, 1]

            correlations.append(
                correlation
            )

    if correlations:

        mean_correlation = np.mean(
            correlations
        )

        print(
            f"Mean tuning correlation: "
            f"{mean_correlation:.3f}"
        )

        print(
            f"Best tuning correlation: "
            f"{max(correlations):.3f}"
        )

        print(
            f"Worst tuning correlation: "
            f"{min(correlations):.3f}"
        )

    else:

        mean_correlation = 0.0

        print(
            "Could not calculate "
            "tuning correlations."
        )

    # ---------------------------------------------------------
    # Plot tuning curves.
    # ---------------------------------------------------------

    plt.figure(
        figsize=(11, 7)
    )

    for rank, index in enumerate(
        selected,
        start=1
    ):

        neuron_id = candidate_neurons[
            index
        ]

        response = mean_activity[
            :,
            index
        ]

        plt.plot(
            POSITIONS,
            response,
            marker="o",
            label=f"Neuron {neuron_id}"
        )

    plt.xlabel(
        "Visual position"
    )

    plt.ylabel(
        "Average spikes"
    )

    plt.title(
        "FLY-001 Position-Tuned Neurons"
    )

    plt.grid(True)

    plt.legend(
        fontsize=8
    )

    plt.tight_layout()

    plt.savefig(
        "fly_position_tuning.png",
        dpi=150
    )

    plt.show()

    # ---------------------------------------------------------
    # Save data.
    # ---------------------------------------------------------

    np.savez(
        "fly_position_tuning.npz",
        positions=POSITIONS,
        candidate_neurons=candidate_neurons,
        activity=activity,
        mean_activity=mean_activity,
        tuning_strength=tuning_strength,
        preferred_positions=preferred_positions
    )

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        "fly_position_tuning.npz"
    )

    print(
        "fly_position_tuning.png"
    )

    print()
    print("=" * 70)
    print("POSITION TUNING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()