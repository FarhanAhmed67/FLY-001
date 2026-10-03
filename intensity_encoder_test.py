import numpy as np

from flybrain import FlyBrain
from fly_encoder import FlyEncoder


brain = FlyBrain(device="auto")

encoder = FlyEncoder(
    brain.azimuth
)


print()
print("=" * 60)
print("FLY-001 INTENSITY ENCODER TEST")
print("=" * 60)
print()


for name, value in encoder.intensities.items():

    stimulus = encoder.encode_intensity(
        name
    )

    active = np.sum(
        stimulus > 0
    )

    total_input = np.sum(
        stimulus
    )

    print(
        f"{name:8s} | "
        f"intensity: {value:.2f} | "
        f"active receptors: {active:3d} | "
        f"total input: {total_input:.1f}"
    )


print()
print("Position + intensity encoder ready.")