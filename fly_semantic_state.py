class FlySemanticState:

    def __init__(
        self,
        concept,
        direction=None,
        intensity="BRIGHT",
        temporal="STATIC"
    ):

        self.concept = concept
        self.direction = direction
        self.intensity = intensity
        self.temporal = temporal

    def describe(self):

        print()
        print("INTERNAL FLY STATE")
        print("-" * 40)

        print(
            f"Concept   : {self.concept}"
        )

        print(
            f"Direction : {self.direction}"
        )

        print(
            f"Intensity : {self.intensity}"
        )

        print(
            f"Temporal  : {self.temporal}"
        )

        print("-" * 40)

    def to_dict(self):

        return {
            "concept": self.concept,
            "direction": self.direction,
            "intensity": self.intensity,
            "temporal": self.temporal
        }