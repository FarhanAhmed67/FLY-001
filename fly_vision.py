import numpy as np
from fly_core import Fly001
from fly_state import FlyState


class FlyVision:

    def __init__(self, fly):
        self.fly = fly
        self.brain = fly.brain
        self.n = fly.n_visual

        self.azimuth = np.asarray(
            self.brain.azimuth,
            dtype=np.float32
        )

    def directional_light(
        self,
        direction=0.0,
        width=0.15,
        intensity=1.0
    ):
        direction = np.clip(
            direction,
            -1.0,
            1.0
        )

        width = max(
            0.01,
            width
        )

        intensity = np.clip(
            intensity,
            0.0,
            1.0
        )

        distance = np.abs(
            self.azimuth - direction
        )

        visual = np.zeros(
            self.n,
            dtype=np.float32
        )

        mask = distance <= width

        visual[mask] = intensity

        return visual


if __name__ == "__main__":

    fly = Fly001()
    vision = FlyVision(fly)
    state = FlyState(fly)

    directions = np.linspace(
        -1.0,
        1.0,
        9
    )

    trials = 20

    print("FLY-001 directional response mapping")
    print("Directions:", directions)
    print("Trials per direction:", trials)

    for direction in directions:

        spike_values = []

        steering_left = 0
        steering_right = 0

        forward_left = 0
        forward_right = 0

        backward_left = 0
        backward_right = 0

        escape_left = 0
        escape_right = 0

        punch_left = 0
        punch_right = 0

        kick_left = 0
        kick_right = 0

        visual = vision.directional_light(
            direction=direction,
            width=0.15,
            intensity=1.0
        )

        active = int(
            np.sum(visual > 0)
        )

        for _ in range(trials):

            fly.reset()

            result = state.run(
                visual,
                steps=20
            )

            spike_values.append(
                result["total_spikes"]
            )

            steering_left += result["steering"]["left"]
            steering_right += result["steering"]["right"]

            forward_left += result["movement"]["forward_left"]
            forward_right += result["movement"]["forward_right"]

            backward_left += result["movement"]["backward_left"]
            backward_right += result["movement"]["backward_right"]

            escape_left += result["escape"]["left"]
            escape_right += result["escape"]["right"]

            punch_left += result["other"]["punch_left"]
            punch_right += result["other"]["punch_right"]

            kick_left += result["other"]["kick_left"]
            kick_right += result["other"]["kick_right"]

        print(
            f"\nDirection {direction:+.2f}"
        )

        print(
            "Active receptors:",
            active
        )

        print(
            "Average spikes:",
            f"{np.mean(spike_values):.1f}"
        )

        print(
            "Steering:",
            f"L={steering_left / trials:.2f}",
            f"R={steering_right / trials:.2f}"
        )

        print(
            "Movement:",
            f"FL={forward_left / trials:.2f}",
            f"FR={forward_right / trials:.2f}",
            f"BL={backward_left / trials:.2f}",
            f"BR={backward_right / trials:.2f}"
        )

        print(
            "Escape:",
            f"L={escape_left / trials:.2f}",
            f"R={escape_right / trials:.2f}"
        )

        print(
            "Other:",
            f"PL={punch_left / trials:.2f}",
            f"PR={punch_right / trials:.2f}",
            f"KL={kick_left / trials:.2f}",
            f"KR={kick_right / trials:.2f}"
        )

    fly = Fly001()
    vision = FlyVision(fly)
    state = FlyState(fly)

    print("FLY-001 directional vision initialized")
    print("Visual receptors:", vision.n)
    print("Azimuth range:",
          vision.azimuth.min(),
          "to",
          vision.azimuth.max())

    directions = [
        -0.8,
        -0.4,
        0.4,
        0.8
    ]

    for direction in directions:

        fly.reset()

        visual = vision.directional_light(
            direction=direction,
            width=0.15,
            intensity=1.0
        )

        result = state.run(
            visual,
            steps=20
        )

        print(
            f"\n========== DIRECTION {direction:+.2f} =========="
        )

        print(
            "Active receptors:",
            int(np.sum(visual > 0))
        )

        print(
            "Total neural spikes:",
            result["total_spikes"]
        )

        print(
            "Movement:",
            result["movement"]
        )

        print(
            "Steering:",
            result["steering"]
        )

        print(
            "Escape:",
            result["escape"]
        )

        print(
            "Other:",
            result["other"]
        )