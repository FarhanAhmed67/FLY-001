import numpy as np


class FlyEncoder:

    def __init__(self, azimuth, n_receptors=500):

        self.azimuth = np.asarray(azimuth)
        self.n_receptors = n_receptors

        self.positions = {
            "LEFT": -0.75,
            "CENTER_LEFT": -0.375,
            "CENTER": 0.0,
            "CENTER_RIGHT": 0.375,
            "RIGHT": 0.75,
        }

        self.intensities = {
            "DARK": 0.0,
            "DIM": 0.25,
            "MEDIUM": 0.5,
            "BRIGHT": 1.0,
        }

    def encode_position(
        self,
        position,
        intensity=1.0
    ):

        if position not in self.positions:

            raise ValueError(
                f"Unknown position: {position}"
            )

        direction = self.positions[position]

        distance = np.abs(
            self.azimuth - direction
        )

        valid = np.where(
            np.isfinite(distance)
        )[0]

        selected = valid[
            np.argsort(distance[valid])[
                :self.n_receptors
            ]
        ]

        stimulus = np.zeros(
            len(self.azimuth),
            dtype=np.float32
        )

        stimulus[selected] = intensity

        return stimulus

    def encode(
        self,
        position,
        intensity=1.0
    ):

        return self.encode_position(
            position,
            intensity
        )

    def encode_intensity(
        self,
        intensity_name
    ):

        if intensity_name not in self.intensities:

            raise ValueError(
                f"Unknown intensity: "
                f"{intensity_name}"
            )

        intensity = self.intensities[
            intensity_name
        ]

        return self.encode_position(
            "CENTER",
            intensity
        )