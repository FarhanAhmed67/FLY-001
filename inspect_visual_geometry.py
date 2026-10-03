import numpy as np
from flybrain import FlyBrain


print()
print("=" * 65)
print("FLY-001 VISUAL RECEPTOR GEOMETRY")
print("=" * 65)

brain = FlyBrain(device="auto")

print()
print("Total neurons       :", len(brain.cell_type))
print("Visual receptors    :", len(brain.visual))
print("Azimuth values      :", len(brain.azimuth))

print()
print("Checking available brain attributes...")

for name in dir(brain):
    if any(word in name.lower() for word in [
        "visual",
        "position",
        "azimuth",
        "elevation",
        "angle",
        "direction"
    ]):
        if not name.startswith("_"):
            print(" ", name)


print()
print("=" * 65)
print("VISUAL RECEPTOR POSITIONS")
print("=" * 65)

try:

    visual_positions = np.asarray(
        brain.positions[brain.visual]
    )

    print()
    print("Shape:", visual_positions.shape)

    print()
    print("First 10 receptor positions:")

    for i in range(
        min(10, len(visual_positions))
    ):
        print(
            f"{i:4d}: "
            f"x={visual_positions[i,0]:+.5f} "
            f"y={visual_positions[i,1]:+.5f} "
            f"z={visual_positions[i,2]:+.5f}"
        )

    print()
    print("Finite values by coordinate:")

    for axis, name in enumerate(
        ["X", "Y", "Z"]
    ):

        values = visual_positions[:, axis]

        finite = np.isfinite(values)

        print(
            f"{name}: "
            f"{finite.sum()} / "
            f"{len(values)} finite"
        )

        if np.any(finite):

            print(
                f"    min = {values[finite].min():+.5f}"
            )

            print(
                f"    max = {values[finite].max():+.5f}"
            )

            print(
                f"    mean = {values[finite].mean():+.5f}"
            )


except Exception as e:

    print()
    print("Could not map visual receptor positions.")

    print(
        "Error:",
        repr(e)
    )


print()
print("=" * 65)
print("AZIMUTH CHECK")
print("=" * 65)

azimuth = np.asarray(
    brain.azimuth
)

print(
    "Min       :",
    azimuth.min()
)

print(
    "Max       :",
    azimuth.max()
)

print(
    "Unique    :",
    len(np.unique(azimuth))
)

print(
    "All finite:",
    np.all(np.isfinite(azimuth))
)


print()
print("=" * 65)
print("DONE")
print("=" * 65)