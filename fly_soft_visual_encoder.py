import numpy as np
import matplotlib.pyplot as plt
from flybrain import FlyBrain

POSITIONS = np.linspace(-1.0, 1.0, 201)

SIGMA = 0.08


def soft_encode(azimuth, position, sigma):
    azimuth = np.asarray(azimuth, dtype=np.float32)

    distance = np.abs(azimuth - position)

    stimulus = np.exp(
        -(distance ** 2) / (2.0 * sigma ** 2)
    )

    stimulus[~np.isfinite(azimuth)] = 0.0

    maximum = stimulus.max()

    if maximum > 0:
        stimulus = stimulus / maximum

    return stimulus.astype(np.float32)


def main():
    print()
    print("=" * 60)
    print("FLY-001 SOFT VISUAL ENCODER")
    print("=" * 60)

    brain = FlyBrain(device="auto")

    azimuth = brain.azimuth

    print()
    print("Visual receptors :", len(azimuth))
    print("Positions tested :", len(POSITIONS))
    print("Sigma            :", SIGMA)

    stimuli = []

    for position in POSITIONS:
        stimulus = soft_encode(
            azimuth,
            position,
            SIGMA
        )

        stimuli.append(stimulus)

    stimuli = np.asarray(stimuli)

    print()
    print("=" * 60)
    print("ADJACENT STIMULUS SIMILARITY")
    print("=" * 60)
    print()

    similarities = []

    for i in range(len(POSITIONS) - 1):
        a = stimuli[i]
        b = stimuli[i + 1]

        numerator = np.dot(a, b)

        denominator = (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )

        similarity = numerator / denominator

        similarities.append(similarity)

        print(
            f"{POSITIONS[i]:+.2f} -> "
            f"{POSITIONS[i + 1]:+.2f} : "
            f"{similarity * 100:.2f}%"
        )

    similarities = np.asarray(similarities)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()

    print(
        f"Mean adjacent similarity : "
        f"{similarities.mean() * 100:.2f}%"
    )

    print(
        f"Minimum similarity       : "
        f"{similarities.min() * 100:.2f}%"
    )

    print(
        f"Maximum similarity       : "
        f"{similarities.max() * 100:.2f}%"
    )

    print(
        f"Similarity variation     : "
        f"{similarities.std() * 100:.2f}%"
    )

    print()
    print("=" * 60)
    print("STIMULUS STATISTICS")
    print("=" * 60)
    print()

    mean_activity = stimuli.mean(axis=1)
    active_receptors = (stimuli > 0.1).sum(axis=1)

    print(
        f"Mean receptor activity  : "
        f"{mean_activity.mean():.4f}"
    )

    print(
        f"Mean active receptors   : "
        f"{active_receptors.mean():.1f}"
    )

    print(
        f"Min active receptors    : "
        f"{active_receptors.min()}"
    )

    print(
        f"Max active receptors    : "
        f"{active_receptors.max()}"
    )

    np.savez(
        "fly_soft_visual_encoder.npz",
        positions=POSITIONS,
        stimuli=stimuli,
        similarities=similarities,
        sigma=SIGMA
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        POSITIONS[:-1],
        similarities * 100
    )

    plt.xlabel(
        "Visual position"
    )

    plt.ylabel(
        "Adjacent cosine similarity (%)"
    )

    plt.title(
        "FLY-001 Soft Visual Encoder Smoothness"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "fly_soft_encoder_smoothness.png",
        dpi=150
    )

    print()
    print("=" * 60)
    print("FILES SAVED")
    print("=" * 60)
    print()

    print(
        "fly_soft_visual_encoder.npz"
    )

    print(
        "fly_soft_encoder_smoothness.png"
    )

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()