import numpy as np
import matplotlib.pyplot as plt

from flybrain import FlyBrain


POSITIONS = np.linspace(-1.0, 1.0, 21)

TRIALS = 10
STEPS = 50


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
    print("FLY-001 POPULATION RECEPTIVE FIELD MAP")
    print("=" * 70)

    print()
    print(
        f"Positions : {len(POSITIONS)}"
    )

    print(
        f"Trials    : {TRIALS}"
    )

    print(
        f"Steps     : {STEPS}"
    )

    # --------------------------------------------------
    # LOAD TUNED NEURONS
    # --------------------------------------------------

    model = np.load(
        "fly_tuned_position_decoder.npz"
    )

    tuned_neurons = model[
        "tuned_indices"
    ]

    print()
    print(
        f"Loaded {len(tuned_neurons)} "
        "tuned neurons."
    )

    print()
    print("Neuron IDs:")

    print(
        " ".join(
            str(int(x))
            for x in tuned_neurons
        )
    )

    # --------------------------------------------------
    # INITIALIZE BRAIN
    # --------------------------------------------------

    brain = FlyBrain(
        device="auto"
    )

    # --------------------------------------------------
    # COLLECT ACTIVITY
    # --------------------------------------------------

    activity = np.zeros(
        (
            len(POSITIONS),
            TRIALS,
            len(tuned_neurons)
        ),
        dtype=np.float32
    )

    for p_index, position in enumerate(
        POSITIONS
    ):

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
                tuned_neurons
            ]

        print(
            "  Mean tuned-neuron activity: "
            f"{activity[p_index].sum(axis=1).mean():.1f}"
        )

    # --------------------------------------------------
    # AVERAGE ACROSS TRIALS
    # --------------------------------------------------

    mean_activity = activity.mean(
        axis=1
    )

    # --------------------------------------------------
    # PRINT RECEPTIVE FIELD SUMMARY
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("RECEPTIVE FIELD SUMMARY")
    print("=" * 70)

    print()

    print(
        f"{'Neuron':<10}"
        f"{'Preferred':<12}"
        f"{'Peak':<10}"
        f"{'Baseline':<12}"
        f"{'Selectivity':<12}"
    )

    print("-" * 70)

    preferred_positions = []
    selectivity_values = []

    for n, neuron_id in enumerate(
        tuned_neurons
    ):

        response = mean_activity[
            :,
            n
        ]

        peak_index = np.argmax(
            response
        )

        peak = response[
            peak_index
        ]

        baseline = response.min()

        selectivity = (
            peak - baseline
        ) / max(
            peak,
            1.0
        )

        preferred = POSITIONS[
            peak_index
        ]

        preferred_positions.append(
            preferred
        )

        selectivity_values.append(
            selectivity
        )

        print(
            f"{int(neuron_id):<10}"
            f"{preferred:+.2f}"
            f"{'':<8}"
            f"{peak:<10.1f}"
            f"{baseline:<12.1f}"
            f"{selectivity:<12.3f}"
        )

    # --------------------------------------------------
    # POPULATION COVERAGE
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("POPULATION COVERAGE")
    print("=" * 70)

    print()

    for position in POSITIONS:

        count = sum(
            abs(
                np.asarray(
                    preferred_positions
                ) - position
            ) < 0.11
        )

        print(
            f"{position:+.2f}"
            f" : {count} neurons"
        )

    print()

    print(
        "Mean selectivity: "
        f"{np.mean(selectivity_values):.3f}"
    )

    print(
        "Maximum selectivity: "
        f"{np.max(selectivity_values):.3f}"
    )

    # --------------------------------------------------
    # CORRELATION MATRIX
    # --------------------------------------------------

    correlation_matrix = np.corrcoef(
        mean_activity.T
    )

    print()
    print("=" * 70)
    print("NEURON CORRELATION")
    print("=" * 70)

    upper = correlation_matrix[
        np.triu_indices(
            len(tuned_neurons),
            k=1
        )
    ]

    print()

    print(
        f"Mean pairwise correlation: "
        f"{np.mean(upper):.3f}"
    )

    print(
        f"Minimum pairwise correlation: "
        f"{np.min(upper):.3f}"
    )

    print(
        f"Maximum pairwise correlation: "
        f"{np.max(upper):.3f}"
    )

    # --------------------------------------------------
    # SAVE DATA
    # --------------------------------------------------

    np.savez(
        "fly_population_receptive_fields.npz",

        positions=POSITIONS,

        tuned_neurons=tuned_neurons,

        activity=activity,

        mean_activity=mean_activity,

        preferred_positions=np.asarray(
            preferred_positions
        ),

        selectivity=np.asarray(
            selectivity_values
        ),

        correlation_matrix=
            correlation_matrix
    )

    # --------------------------------------------------
    # PLOT 1: INDIVIDUAL RECEPTIVE FIELDS
    # --------------------------------------------------

    plt.figure(
        figsize=(12, 8)
    )

    for n, neuron_id in enumerate(
        tuned_neurons
    ):

        plt.plot(
            POSITIONS,
            mean_activity[:, n],
            marker="o",
            label=str(int(neuron_id))
        )

    plt.xlabel(
        "Visual position"
    )

    plt.ylabel(
        "Average spikes"
    )

    plt.title(
        "FLY-001 Tuned Neuron Receptive Fields"
    )

    plt.grid(True)

    plt.legend(
        fontsize=7,
        ncol=2
    )

    plt.tight_layout()

    plt.savefig(
        "fly_population_receptive_fields.png",
        dpi=150
    )

    # --------------------------------------------------
    # PLOT 2: POPULATION HEATMAP
    # --------------------------------------------------

    plt.figure(
        figsize=(12, 8)
    )

    plt.imshow(
        mean_activity.T,
        aspect="auto",
        origin="lower",
        extent=[
            POSITIONS[0],
            POSITIONS[-1],
            0,
            len(tuned_neurons)
        ]
    )

    plt.xlabel(
        "Visual position"
    )

    plt.ylabel(
        "Tuned neuron index"
    )

    plt.title(
        "FLY-001 Population Receptive Field Map"
    )

    plt.colorbar(
        label="Average spikes"
    )

    plt.tight_layout()

    plt.savefig(
        "fly_population_receptive_field_heatmap.png",
        dpi=150
    )

    plt.show()

    # --------------------------------------------------
    # COMPLETE
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        "fly_population_receptive_fields.npz"
    )

    print(
        "fly_population_receptive_fields.png"
    )

    print(
        "fly_population_receptive_field_heatmap.png"
    )

    print()
    print("=" * 70)
    print("POPULATION RECEPTIVE FIELD MAP COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()