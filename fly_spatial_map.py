import numpy as np
import matplotlib.pyplot as plt

from flybrain import FlyBrain


POSITIONS = np.linspace(-1.0, 1.0, 21)

TRIALS = 10


def generate_sample(brain, position):

    valid = np.where(
        np.isfinite(brain.azimuth)
    )[0]

    distance = np.abs(
        brain.azimuth[valid]
        - position
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

    total_spikes = 0

    for _ in range(5):

        for _ in range(10):

            fired = brain.step(
                eye_drive=visual
            )

            total_spikes += len(fired)

    return total_spikes


def main():

    print()
    print("=" * 60)
    print("FLY-001 SPATIAL NEURAL RESPONSE MAP")
    print("=" * 60)

    brain = FlyBrain(
        device="auto"
    )

    averages = []

    for index, position in enumerate(
        POSITIONS
    ):

        print()
        print(
            f"Position {index + 1}/"
            f"{len(POSITIONS)}: "
            f"{position:+.2f}"
        )

        trials = []

        for trial in range(TRIALS):

            spikes = generate_sample(
                brain,
                position
            )

            trials.append(spikes)

            print(
                f"  Trial {trial + 1}: "
                f"{spikes}"
            )

        average = np.mean(trials)

        averages.append(average)

        print(
            f"  Average: "
            f"{average:.1f}"
        )

    averages = np.asarray(
        averages
    )

    print()
    print("=" * 60)
    print("SPATIAL RESPONSE SUMMARY")
    print("=" * 60)

    for position, average in zip(
        POSITIONS,
        averages
    ):

        print(
            f"{position:+.2f} "
            f"→ {average:.1f} spikes"
        )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        POSITIONS,
        averages,
        marker="o"
    )

    plt.xlabel(
        "Visual position"
    )

    plt.ylabel(
        "Average neural spikes"
    )

    plt.title(
        "FLY-001 Spatial Neural Response"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        "fly_spatial_map.png",
        dpi=150
    )

    plt.show()

    print()
    print(
        "Saved: fly_spatial_map.png"
    )


if __name__ == "__main__":

    main()