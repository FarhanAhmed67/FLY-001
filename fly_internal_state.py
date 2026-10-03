class FlyInternalState:

    ACTIVITY_REFERENCE = 500000.0

    def __init__(self):

        self.current_direction = None
        self.previous_direction = None

        self.direction_changes = 0

        self.activity_raw = 0.0
        self.activity_level = 0.0

        self.stability = 0.0
        self.stability_label = "LOW"

        self.recent_transition = (
            "INSUFFICIENT_DATA"
        )

        self.pattern = (
            "INSUFFICIENT_DATA"
        )

        self.interaction_count = 0

        self.recent_directions = []

        self.internal_drive = 1.0

    def update(
        self,
        memory_summary,
        neural_activity=0.0
    ):

        current = (
            memory_summary.get(
                "current_state"
            )
        )

        previous = (
            memory_summary.get(
                "previous_state"
            )
        )

        if current is not None:

            self.current_direction = (
                current.get(
                    "direction"
                )
            )

        else:

            self.current_direction = None

        if previous is not None:

            self.previous_direction = (
                previous.get(
                    "direction"
                )
            )

        else:

            self.previous_direction = None

        self.direction_changes = (
            memory_summary.get(
                "direction_changes",
                0
            )
        )

        self.pattern = (
            memory_summary.get(
                "pattern",
                "INSUFFICIENT_DATA"
            )
        )

        self.interaction_count = (
            memory_summary.get(
                "interactions",
                0
            )
        )

        self.recent_directions = (
            memory_summary.get(
                "recent_directions",
                []
            )
        )

        self.activity_raw = float(
            neural_activity
        )

        self.activity_level = (
            self.activity_raw
            / self.ACTIVITY_REFERENCE
        )

        self.activity_level = max(
            0.0,
            min(
                1.0,
                self.activity_level
            )
        )

        self.detect_recent_transition()

        self.calculate_stability()

        self.calculate_internal_drive()

    def detect_recent_transition(self):

        if (
            self.previous_direction
            is None
            or self.current_direction
            is None
        ):

            self.recent_transition = (
                "INSUFFICIENT_DATA"
            )

            return

        if (
            self.current_direction
            == self.previous_direction
        ):

            self.recent_transition = (
                "STABLE"
            )

        else:

            self.recent_transition = (
                "CHANGED"
            )

    def calculate_stability(self):

        directions = (
            self.recent_directions
        )

        if len(directions) < 2:

            self.stability = 0.0
            self.stability_label = "LOW"

            return

        total_transitions = (
            len(directions) - 1
        )

        changes = (
            self.direction_changes
        )

        self.stability = (
            1.0
            - (
                changes
                / total_transitions
            )
        )

        self.stability = max(
            0.0,
            min(
                1.0,
                self.stability
            )
        )

        if self.stability >= 0.75:

            self.stability_label = (
                "HIGH"
            )

        elif self.stability >= 0.40:

            self.stability_label = (
                "MEDIUM"
            )

        else:

            self.stability_label = (
                "LOW"
            )

    def calculate_internal_drive(self):

        drive = 1.0

        if self.pattern == "STABLE":

            drive += 0.10

        elif self.pattern == "CHANGING":

            drive -= 0.05

        elif self.pattern == "ALTERNATING":

            drive -= 0.10

        if self.recent_transition == "STABLE":

            drive += 0.05

        elif self.recent_transition == "CHANGED":

            drive -= 0.05

        drive += (
            self.activity_level - 0.5
        ) * 0.10

        self.internal_drive = max(
            0.80,
            min(
                1.20,
                drive
            )
        )

    def get_activity_label(self):

        if self.activity_level >= 0.80:

            return "HIGH"

        if self.activity_level >= 0.40:

            return "MEDIUM"

        return "LOW"

    def get_internal_drive(self):

        return self.internal_drive

    def to_dict(self):

        return {

            "current_direction":
                self.current_direction,

            "previous_direction":
                self.previous_direction,

            "direction_changes":
                self.direction_changes,

            "activity_raw":
                self.activity_raw,

            "activity_level":
                self.activity_level,

            "activity_label":
                self.get_activity_label(),

            "stability":
                self.stability,

            "stability_label":
                self.stability_label,

            "recent_transition":
                self.recent_transition,

            "pattern":
                self.pattern,

            "interaction_count":
                self.interaction_count,

            "recent_directions":
                self.recent_directions,

            "internal_drive":
                self.internal_drive
        }

    def describe(self):

        print()
        print(
            "FLY INTERNAL STATE"
        )

        print(
            "-" * 40
        )

        print(
            f"Current direction : "
            f"{self.current_direction}"
        )

        print(
            f"Previous direction: "
            f"{self.previous_direction}"
        )

        print(
            f"Recent transition : "
            f"{self.recent_transition}"
        )

        print(
            f"Direction changes : "
            f"{self.direction_changes}"
        )

        print(
            f"Pattern           : "
            f"{self.pattern}"
        )

        print(
            f"Stability         : "
            f"{self.stability:.2f}"
        )

        print(
            f"Stability level   : "
            f"{self.stability_label}"
        )

        print(
            f"Neural activity   : "
            f"{self.activity_level:.2f}"
        )

        print(
            f"Activity level    : "
            f"{self.get_activity_label()}"
        )

        print(
            f"Raw neural spikes : "
            f"{self.activity_raw:.0f}"
        )

        print(
            f"Internal drive    : "
            f"{self.internal_drive:.3f}"
        )

        print(
            f"Interactions      : "
            f"{self.interaction_count}"
        )

        if self.recent_directions:

            print(
                "Recent directions : "
                + " -> ".join(
                    self.recent_directions
                )
            )

        else:

            print(
                "Recent directions : None"
            )

        print(
            "-" * 40
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

        "direction_changes": 2,

        "pattern": "CHANGING",

        "interactions": 9,

        "recent_directions": [
            "RIGHT",
            "CENTER",
            "LEFT",
            "LEFT",
            "LEFT"
        ]
    }

    state.update(
        test_memory,
        neural_activity=448309
    )

    state.describe()