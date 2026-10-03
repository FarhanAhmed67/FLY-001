import numpy as np
from flybrain import FlyBrain
from fly_encoder import FlyEncoder


brain = FlyBrain(device="auto")

encoder = FlyEncoder(
    brain.azimuth
)


print()
print("=" * 60)
print("FLY-001 ENCODER TEST")
print("=" * 60)

for concept in encoder.concepts:

    stimulus = encoder.encode(concept)

    active = np.sum(stimulus > 0)

    print(
        f"{concept:12s} -> "
        f"{active} active receptors"
    )

print()
print("Encoder ready.")