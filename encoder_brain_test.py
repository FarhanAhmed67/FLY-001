import numpy as np
from flybrain import FlyBrain
from fly_encoder import FlyEncoder


brain = FlyBrain(device="auto")
encoder = FlyEncoder(brain.azimuth)


def run_concept(concept, steps=50):

    brain.reset()

    stimulus = encoder.encode(concept)

    total_spikes = 0

    neuron_activity = np.zeros(
        len(brain.cell_type),
        dtype=np.int32
    )

    for _ in range(steps):

        fired = brain.step(
            eye_drive=stimulus
        )

        fired = np.asarray(
            fired,
            dtype=np.int64
        )

        total_spikes += len(fired)

        if len(fired) > 0:

            neuron_activity += np.bincount(
                fired,
                minlength=len(brain.cell_type)
            )

    return total_spikes, neuron_activity


print()
print("=" * 70)
print("FLY-001 ENCODER → BRAIN TEST")
print("=" * 70)
print()

for concept in encoder.concepts:

    spikes, activity = run_concept(
        concept
    )

    active_neurons = np.sum(
        activity > 0
    )

    print(
        f"{concept:12s} | "
        f"spikes: {spikes:7d} | "
        f"active neurons: {active_neurons:6d}"
    )

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)