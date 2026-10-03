import numpy as np
from fly_core import Fly001
from fly_state import FlyState
from fly_vision import FlyVision


fly = Fly001()
vision = FlyVision(fly)
state = FlyState(fly)

directions = np.linspace(-1.0, 1.0, 9)

intensity = 1.0
width = 0.15

trials = 20
steps_per_position = 5

print("FLY-001 moving visual stimulus")
print("Directions:", directions)
print("Intensity:", intensity)
print("Steps per position:", steps_per_position)
print("Trials:", trials)


for trial in range(trials):

    fly.reset()

    print()
    print(f"========== TRIAL {trial + 1} ==========")

    for direction in directions:

        visual = vision.directional_light(
            direction=direction,
            width=width,
            intensity=intensity
        )

        result = state.run(
            visual,
            steps=steps_per_position
        )

        motor = result["movement"]
        steering = result["steering"]
        escape = result["escape"]

        print(
            f"Dir {direction:+.2f} | "
            f"Spikes {result['total_spikes']:5d} | "
            f"Steer L/R {steering['left']}/{steering['right']} | "
            f"Move "
            f"FL/FR {motor['forward_left']}/{motor['forward_right']} "
            f"BL/BR {motor['backward_left']}/{motor['backward_right']} | "
            f"Escape L/R {escape['left']}/{escape['right']}"
        )