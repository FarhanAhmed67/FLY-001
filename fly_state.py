import numpy as np
from fly_core import Fly001


class FlyState:

    def __init__(self, fly):
        self.fly = fly

    def run(self, visual_input, steps=20):

        result = self.fly.run(
            visual_input,
            steps=steps
        )

        motor = result["motor_activity"]

        return {
            "visual_mean": float(np.mean(visual_input)),
            "visual_min": float(np.min(visual_input)),
            "visual_max": float(np.max(visual_input)),
            "total_spikes": result["total_spikes"],

            "movement": {
                "forward_left": motor["forward_L"],
                "forward_right": motor["forward_R"],
                "backward_left": motor["backward_L"],
                "backward_right": motor["backward_R"],
            },

            "steering": {
                "left": motor["steer_L"],
                "right": motor["steer_R"],
            },

            "escape": {
                "left": motor["escape_L"],
                "right": motor["escape_R"],
            },

            "other": {
                "punch_left": motor["punch_L"],
                "punch_right": motor["punch_R"],
                "kick_left": motor["kick_L"],
                "kick_right": motor["kick_R"],
            }
        }


if __name__ == "__main__":

    fly = Fly001()
    state = FlyState(fly)

    fly.reset()

    dark = np.zeros(
        fly.n_visual,
        dtype=np.float32
    )

    bright = np.ones(
        fly.n_visual,
        dtype=np.float32
    )

    print("\n========== DARK ==========")

    dark_state = state.run(
        dark,
        steps=20
    )

    print(dark_state)

    print("\n========== BRIGHT ==========")

    bright_state = state.run(
        bright,
        steps=20
    )

    print(bright_state)