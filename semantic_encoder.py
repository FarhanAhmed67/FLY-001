import numpy as np

from fly_encoder import FlyEncoder


class SemanticEncoder:

    def __init__(self, azimuth):

        self.visual_encoder = FlyEncoder(
            azimuth
        )

        self.concepts = {
            "HELLO": {
                "position": "CENTER",
                "intensity": 1.0
            },

            "YES": {
                "position": "CENTER_RIGHT",
                "intensity": 1.0
            },

            "NO": {
                "position": "CENTER_LEFT",
                "intensity": 1.0
            },

            "LEFT": {
                "position": "LEFT",
                "intensity": 1.0
            },

            "RIGHT": {
                "position": "RIGHT",
                "intensity": 1.0
            },

            "UP": {
                "position": "CENTER",
                "intensity": 0.75
            },

            "DOWN": {
                "position": "CENTER",
                "intensity": 0.25
            },

            "BRIGHT": {
                "position": "CENTER_RIGHT",
                "intensity": 1.0
            },

            "DARK": {
                "position": "CENTER",
                "intensity": 0.0
            },

            "MOVE": {
                "position": "LEFT",
                "intensity": 0.75
            }
        }


    def encode(self, concept):

        if concept not in self.concepts:

            raise ValueError(
                f"Unknown concept: {concept}"
            )

        config = self.concepts[
            concept
        ]

        stimulus = (
            self.visual_encoder.encode_position(
                config["position"],
                config["intensity"]
            )
        )

        return stimulus


    def get_concepts(self):

        return list(
            self.concepts.keys()
        )