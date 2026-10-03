from fly_internal_state import FlyInternalState


class FlyBehavior:

    def __init__(self):

        self.last_action = None

    def decide(
        self,
        decoded_direction,
        internal_state
    ):

        drive = (
            internal_state.get_internal_drive()
        )

        pattern = (
            internal_state.pattern
        )

        stability = (
            internal_state.stability
        )

        previous = (
            internal_state.previous_direction
        )

        current = (
            internal_state.current_direction
        )

        # Strong stable state
        if (
            pattern == "STABLE"
            and stability >= 0.75
        ):

            if current == "LEFT":

                action = "MOVE_LEFT"

            elif current == "RIGHT":

                action = "MOVE_RIGHT"

            elif current == "CENTER":

                action = "STAY_CENTER"

            else:

                action = (
                    f"MOVE_{decoded_direction}"
                )

        # Alternating state
        elif pattern == "ALTERNATING":

            if current != previous:

                action = (
                    f"TURN_{decoded_direction}"
                )

            else:

                action = "STABILIZE"

        # Changing state
        elif pattern == "CHANGING":

            action = (
                f"RESPOND_{decoded_direction}"
            )

        # Mixed state
        elif pattern == "MIXED":

            if drive >= 1.10:

                action = (
                    f"ACTIVE_{decoded_direction}"
                )

            elif drive <= 0.90:

                action = (
                    f"CAUTIOUS_{decoded_direction}"
                )

            else:

                action = (
                    f"MOVE_{decoded_direction}"
                )

        else:

            action = (
                f"MOVE_{decoded_direction}"
            )

        self.last_action = action

        return action

    def describe(
        self,
        action
    ):

        print()
        print(
            "FLY BEHAVIOR"
        )

        print(
            "-" * 40
        )

        print(
            f"Action           : "
            f"{action}"
        )

        print(
            f"Internal drive   : "
            f"{self._drive:.3f}"
        )

        print(
            f"Pattern          : "
            f"{self._pattern}"
        )

        print(
            f"Stability        : "
            f"{self._stability:.2f}"
        )

        print(
            "-" * 40
        )

    def evaluate(
        self,
        action,
        internal_state
    ):

        self._drive = (
            internal_state.get_internal_drive()
        )

        self._pattern = (
            internal_state.pattern
        )

        self._stability = (
            internal_state.stability
        )

        self.describe(
            action
        )


if __name__ == "__main__":

    state = FlyInternalState()

    test_memory = {

        "current_state": {
            "concept": "MOVE",
            "direction": "LEFT"
        },

        "previous_state": {
            "concept": "MOVE",
            "direction": "LEFT"
        },

        "direction_changes": 0,

        "pattern": "STABLE",

        "interactions": 10,

        "recent_directions": [
            "LEFT",
            "LEFT",
            "LEFT",
            "LEFT",
            "LEFT"
        ]
    }

    state.update(
        test_memory,
        neural_activity=450000
    )

    behavior = FlyBehavior()

    action = behavior.decide(
        "LEFT",
        state
    )

    behavior.evaluate(
        action,
        state
    )