import numpy as np

from flybrain import FlyBrain
from fly_interface import PersistentDecoder
from fly_internal_state import FlyInternalState


MODEL_FILE = "fly_decoder_model.npz"


POSITIONS = {
    "LEFT": -0.75,
    "CENTER_LEFT": -0.375,
    "CENTER": 0.0,
    "CENTER_RIGHT": 0.375,
    "RIGHT": 0.75,
}


SEQUENCES = {

    "A_STABLE_LEFT": [
        "LEFT",
        "LEFT",
        "LEFT",
        "LEFT"
    ],

    "B_ALTERNATING": [
        "LEFT",
        "RIGHT",
        "LEFT",
        "RIGHT"
    ],

    "C_GROUPED": [
        "LEFT",
        "RIGHT",
        "RIGHT",
        "LEFT"
    ]
}


def make_stimulus(
    azimuth,
    position,
    intensity
):

    direction = POSITIONS[
        position
    ]

    distance = np.abs(
        azimuth - direction
    )

    valid = np.where(
        np.isfinite(distance)
    )[0]

    selected = valid[
        np.argsort(
            distance[valid]
        )[:500]
    ]

    stimulus = np.zeros(
        len(azimuth),
        dtype=np.float32
    )

    stimulus[selected] = intensity

    return stimulus


def run_brain(
    brain,
    decoder,
    azimuth,
    position,
    internal_drive,
    steps=50
):

    brain.reset()

    windows = []

    total_spikes = 0

    for start in range(
        0,
        steps,
        10
    ):

        window_activity = np.zeros(
            brain.n,
            dtype=np.float32
        )

        for _ in range(10):

            stimulus = make_stimulus(
                azimuth,
                position,
                internal_drive
            )

            fired = brain.step(
                eye_drive=stimulus
            )

            total_spikes += len(
                fired
            )

            if len(fired) > 0:

                window_activity[
                    fired
                ] += 1

        windows.append(
            window_activity
        )

    neural_state = np.concatenate(
        windows
    )

    decoded, confidence = (
        decoder.predict(
            neural_state.reshape(
                1,
                -1
            )
        )
    )

    return (
        decoded,
        confidence,
        total_spikes
    )


def main():

    print()
    print("=" * 70)
    print(
        "FLY-001 FEEDBACK LOOP EXPERIMENT"
    )
    print("=" * 70)

    print()
    print(
        "Initializing brain..."
    )

    brain = FlyBrain(
        device="auto"
    )

    azimuth = np.asarray(
        brain.azimuth
    )

    decoder = PersistentDecoder()

    print(
        f"Neurons: {brain.n}"
    )

    print(
        f"Visual receptors: "
        f"{len(azimuth)}"
    )

    print()

    for sequence_name, sequence in (
        SEQUENCES.items()
    ):

        print()
        print("=" * 70)

        print(
            f"SEQUENCE: {sequence_name}"
        )

        print(
            " → ".join(sequence)
        )

        print("=" * 70)

        internal_state = (
            FlyInternalState()
        )

        for step_number, position in enumerate(
            sequence,
            start=1
        ):

            drive = (
                internal_state
                .get_internal_drive()
            )

            (
                decoded,
                confidence,
                spikes
            ) = run_brain(
                brain,
                decoder,
                azimuth,
                position,
                drive
            )

            current_state = {
                "current_state": {
                    "concept": "MOVE",
                    "direction": position,
                    "intensity": "BRIGHT",
                    "temporal": "STATIC"
                },

                "previous_state": None,

                "direction_changes": 0,

                "pattern": "INSUFFICIENT_DATA",

                "interactions": step_number,

                "recent_directions": []
            }

            if step_number > 1:

                previous = sequence[
                    step_number - 2
                ]

                current_state[
                    "previous_state"
                ] = {
                    "concept": "MOVE",
                    "direction": previous,
                    "intensity": "BRIGHT",
                    "temporal": "STATIC"
                }

                recent = sequence[
                    max(
                        0,
                        step_number - 5
                    ):
                    step_number
                ]

                changes = 0

                for i in range(
                    1,
                    len(recent)
                ):

                    if (
                        recent[i]
                        != recent[i - 1]
                    ):

                        changes += 1

                current_state[
                    "direction_changes"
                ] = changes

                if len(set(recent)) == 1:

                    pattern = "STABLE"

                elif len(recent) >= 4:

                    alternating = True

                    for i in range(
                        2,
                        len(recent)
                    ):

                        if (
                            recent[i]
                            != recent[i - 2]
                        ):

                            alternating = False
                            break

                    if alternating:

                        pattern = "ALTERNATING"

                    elif changes >= len(
                        recent
                    ) // 2:

                        pattern = "CHANGING"

                    else:

                        pattern = "MIXED"

                else:

                    pattern = "MIXED"

                current_state[
                    "pattern"
                ] = pattern

                current_state[
                    "recent_directions"
                ] = recent

            internal_state.update(
                current_state,
                spikes
            )

            new_drive = (
                internal_state
                .get_internal_drive()
            )

            print()

            print(
                f"Step {step_number}"
            )

            print(
                f"  Input position     : "
                f"{position}"
            )

            print(
                f"  Applied drive      : "
                f"{drive:.3f}"
            )

            print(
                f"  Neural spikes      : "
                f"{spikes}"
            )

            print(
                f"  Decoded position   : "
                f"{decoded}"
            )

            print(
                f"  Decoder similarity : "
                f"{confidence:.4f}"
            )

            print(
                f"  Pattern            : "
                f"{internal_state.pattern}"
            )

            print(
                f"  Stability          : "
                f"{internal_state.stability:.2f}"
            )

            print(
                f"  New internal drive : "
                f"{new_drive:.3f}"
            )

    print()
    print("=" * 70)
    print(
        "EXPERIMENT COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()