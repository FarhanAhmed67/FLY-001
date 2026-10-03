import json
import os
from datetime import datetime


class FlyMemory:

    def __init__(
        self,
        filename="fly_memory.json",
        max_history=20
    ):

        self.filename = filename
        self.max_history = max_history

        self.data = {
            "interaction_count": 0,
            "current_state": None,
            "previous_state": None,
            "history": []
        }

        self.load()

    def load(self):

        if not os.path.exists(
            self.filename
        ):
            return

        try:

            with open(
                self.filename,
                "r",
                encoding="utf-8"
            ) as file:

                self.data = json.load(
                    file
                )

        except (
            json.JSONDecodeError,
            OSError
        ):

            print(
                "Warning: Could not load fly memory."
            )

            print(
                "Starting with fresh memory."
            )

    def save(self):

        with open(
            self.filename,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.data,
                file,
                indent=4
            )

    def update(
        self,
        message,
        state
    ):

        self.data[
            "previous_state"
        ] = self.data[
            "current_state"
        ]

        current = state.to_dict()

        self.data[
            "current_state"
        ] = current

        self.data[
            "interaction_count"
        ] += 1

        self.data[
            "history"
        ].append({

            "interaction":
                self.data[
                    "interaction_count"
                ],

            "message":
                message,

            "state":
                current,

            "timestamp":
                datetime.now().isoformat()

        })

        if len(
            self.data["history"]
        ) > self.max_history:

            self.data["history"] = (
                self.data["history"][
                    -self.max_history:
                ]
            )

        self.save()

    def get_current_state(self):

        return self.data[
            "current_state"
        ]

    def get_previous_state(self):

        return self.data[
            "previous_state"
        ]

    def get_interaction_count(self):

        return self.data[
            "interaction_count"
        ]

    def get_history(self):

        return self.data[
            "history"
        ]

    def get_recent_states(
        self,
        count=5
    ):

        history = self.data[
            "history"
        ]

        recent = history[
            -count:
        ]

        return [
            item["state"]
            for item in recent
        ]

    def get_recent_directions(
        self,
        count=5
    ):

        states = self.get_recent_states(
            count
        )

        return [
            state["direction"]
            for state in states
            if state.get(
                "direction"
            ) is not None
        ]

    def count_direction_changes(
        self,
        count=5
    ):

        directions = (
            self.get_recent_directions(
                count
            )
        )

        if len(directions) < 2:
            return 0

        changes = 0

        for i in range(
            1,
            len(directions)
        ):

            if (
                directions[i]
                != directions[i - 1]
            ):

                changes += 1

        return changes

    def detect_pattern(
        self,
        count=5
    ):

        directions = (
            self.get_recent_directions(
                count
            )
        )

        if len(directions) < 2:

            return "INSUFFICIENT_DATA"

        unique = set(
            directions
        )

        changes = (
            self.count_direction_changes(
                count
            )
        )

        if len(unique) == 1:

            return "STABLE"

        if len(directions) >= 4:

            alternating = True

            for i in range(
                2,
                len(directions)
            ):

                if (
                    directions[i]
                    != directions[i - 2]
                ):

                    alternating = False
                    break

            if alternating:

                return "ALTERNATING"

        if changes >= len(
            directions
        ) // 2:

            return "CHANGING"

        return "MIXED"

    def get_state_summary(
        self,
        count=5
    ):

        directions = (
            self.get_recent_directions(
                count
            )
        )

        changes = (
            self.count_direction_changes(
                count
            )
        )

        pattern = (
            self.detect_pattern(
                count
            )
        )

        return {
            "interactions":
                self.data[
                    "interaction_count"
                ],

            "recent_directions":
                directions,

            "direction_changes":
                changes,

            "pattern":
                pattern,

            "current_state":
                self.data[
                    "current_state"
                ],

            "previous_state":
                self.data[
                    "previous_state"
                ]
        }

    def show_memory(self):

        print()
        print(
            "FLY MEMORY"
        )
        print(
            "-" * 40
        )

        print(
            f"Interactions : "
            f"{self.data['interaction_count']}"
        )

        print(
            f"Current      : "
            f"{self.data['current_state']}"
        )

        print(
            f"Previous     : "
            f"{self.data['previous_state']}"
        )

        print(
            f"History size : "
            f"{len(self.data['history'])}"
        )

        print(
            "-" * 40
        )

    def show_recent_state(
        self,
        count=5
    ):

        summary = (
            self.get_state_summary(
                count
            )
        )

        directions = (
            summary[
                "recent_directions"
            ]
        )

        print()
        print(
            "FLY RECENT STATE"
        )
        print(
            "-" * 40
        )

        print(
            f"Interactions     : "
            f"{summary['interactions']}"
        )

        if directions:

            print(
                "Recent directions: "
                + " -> ".join(
                    directions
                )
            )

        else:

            print(
                "Recent directions: None"
            )

        print(
            f"Direction changes: "
            f"{summary['direction_changes']}"
        )

        print(
            f"Pattern          : "
            f"{summary['pattern']}"
        )

        print(
            "-" * 40
        )


if __name__ == "__main__":

    memory = FlyMemory()

    memory.show_memory()

    memory.show_recent_state()