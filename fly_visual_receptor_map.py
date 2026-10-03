import numpy as np
import matplotlib.pyplot as plt
from flybrain import FlyBrain

POSITIONS = np.linspace(-1.0, 1.0, 21)
N_RECEPTORS = 500


def create_receptor_set(brain, position):
    valid = np.where(np.isfinite(brain.azimuth))[0]
    distance = np.abs(brain.azimuth[valid] - position)
    selected = valid[np.argsort(distance)[:N_RECEPTORS]]
    return selected


def main():
    print()
    print("=" * 60)
    print("FLY-001 VISUAL RECEPTOR MAP")
    print("=" * 60)

    brain = FlyBrain(device="auto")

    valid = np.isfinite(brain.azimuth)

    print()
    print("Total visual receptors :", len(brain.azimuth))
    print("Valid receptors        :", int(valid.sum()))
    print("Unique azimuth values  :", len(np.unique(brain.azimuth[valid])))
    print("Positions tested       :", len(POSITIONS))
    print("Receptors per position :", N_RECEPTORS)

    receptor_sets = []

    for position in POSITIONS:
        selected = create_receptor_set(brain, position)
        receptor_sets.append(selected)

    overlap_matrix = np.zeros(
        (len(POSITIONS), len(POSITIONS)),
        dtype=np.float32
    )

    for i in range(len(POSITIONS)):
        set_a = set(receptor_sets[i])

        for j in range(len(POSITIONS)):
            set_b = set(receptor_sets[j])

            shared = len(set_a & set_b)

            overlap_matrix[i, j] = shared / N_RECEPTORS

    print()
    print("=" * 60)
    print("ADJACENT POSITION OVERLAP")
    print("=" * 60)
    print()

    adjacent_overlap = []

    for i in range(len(POSITIONS) - 1):
        overlap = overlap_matrix[i, i + 1]
        adjacent_overlap.append(overlap)

        changed = N_RECEPTORS - int(overlap * N_RECEPTORS)

        print(
            f"{POSITIONS[i]:+.2f} -> "
            f"{POSITIONS[i + 1]:+.2f} : "
            f"{overlap * 100:.1f}% overlap | "
            f"{changed} changed"
        )

    print()
    print("=" * 60)
    print("POSITION DISTANCE VS OVERLAP")
    print("=" * 60)
    print()

    for separation in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        values = []

        for i in range(len(POSITIONS)):
            for j in range(i + 1, len(POSITIONS)):
                distance = abs(POSITIONS[i] - POSITIONS[j])

                if abs(distance - separation) < 0.001:
                    values.append(overlap_matrix[i, j])

        if values:
            print(
                f"Separation {separation:.1f} : "
                f"mean {np.mean(values) * 100:.1f}% | "
                f"min {np.min(values) * 100:.1f}% | "
                f"max {np.max(values) * 100:.1f}%"
            )

    all_receptors = np.concatenate(receptor_sets)
    unique_receptors = np.unique(all_receptors)

    total_selections = len(POSITIONS) * N_RECEPTORS

    reuse_ratio = len(unique_receptors) / total_selections

    print()
    print("=" * 60)
    print("RECEPTOR COVERAGE")
    print("=" * 60)
    print()

    print("Total selections      :", total_selections)
    print("Unique receptors used :", len(unique_receptors))
    print(f"Reuse ratio           : {reuse_ratio:.3f}")

    unique_sets = set()

    for receptors in receptor_sets:
        unique_sets.add(tuple(receptors.tolist()))

    print()
    print("=" * 60)
    print("ENCODING UNIQUENESS")
    print("=" * 60)
    print()

    print("Positions tested     :", len(POSITIONS))
    print("Unique receptor sets :", len(unique_sets))

    if len(unique_sets) < len(POSITIONS):
        print("WARNING: Some positions use the exact same receptor set.")
    else:
        print("Every position has a unique receptor set.")

    unique_azimuth, counts = np.unique(
        brain.azimuth[valid],
        return_counts=True
    )

    print()
    print("=" * 60)
    print("AZIMUTH DISTRIBUTION")
    print("=" * 60)
    print()

    print(f"Minimum azimuth    : {unique_azimuth.min():+.3f}")
    print(f"Maximum azimuth    : {unique_azimuth.max():+.3f}")
    print(f"Number of azimuths : {len(unique_azimuth)}")
    print(f"Mean receptors/bin : {counts.mean():.1f}")
    print(f"Min receptors/bin  : {counts.min()}")
    print(f"Max receptors/bin  : {counts.max()}")

    np.savez(
        "fly_visual_receptor_map.npz",
        positions=POSITIONS,
        receptor_sets=np.asarray(receptor_sets),
        overlap_matrix=overlap_matrix,
        unique_azimuth=unique_azimuth,
        azimuth_counts=counts
    )

    plt.figure(figsize=(9, 8))

    plt.imshow(
        overlap_matrix,
        origin="lower",
        extent=[
            POSITIONS[0],
            POSITIONS[-1],
            POSITIONS[0],
            POSITIONS[-1]
        ],
        aspect="auto",
        vmin=0.0,
        vmax=1.0
    )

    plt.xlabel("Visual position")
    plt.ylabel("Visual position")
    plt.title("FLY-001 Visual Receptor Overlap")

    plt.colorbar(
        label="Fraction of shared receptors"
    )

    plt.tight_layout()

    plt.savefig(
        "fly_visual_receptor_overlap.png",
        dpi=150
    )

    plt.figure(figsize=(10, 6))

    midpoint = POSITIONS[:-1] + 0.05

    plt.plot(
        midpoint,
        np.asarray(adjacent_overlap) * 100,
        marker="o"
    )

    plt.xlabel("Position midpoint")
    plt.ylabel("Adjacent overlap (%)")
    plt.title("FLY-001 Adjacent Visual Position Overlap")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "fly_visual_adjacent_overlap.png",
        dpi=150
    )

    print()
    print("=" * 60)
    print("FILES SAVED")
    print("=" * 60)
    print()

    print("fly_visual_receptor_map.npz")
    print("fly_visual_receptor_overlap.png")
    print("fly_visual_adjacent_overlap.png")

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
