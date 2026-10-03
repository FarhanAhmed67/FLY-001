import numpy as np
from fly_core import Fly001
from fly_state import FlyState
from fly_vision import FlyVision


fly = Fly001()
vision = FlyVision(fly)
state = FlyState(fly)

directions = np.linspace(-1.0, 1.0, 9)

target_receptors = 500
intensity = 1.0
trials = 20
steps = 20

print("FLY-001 equalized spatial response mapping")
print("Target receptors:", target_receptors)
print("Intensity:", intensity)
print("Trials per direction:", trials)


for direction in directions:

    distance = np.abs(
        vision.azimuth - direction
    )

    order = np.argsort(distance)

    selected = order[:target_receptors]

    visual = np.zeros(
        vision.n,
        dtype=np.float32
    )

    visual[selected] = intensity

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

    for _ in range(trials):

        fly.reset()

        result = state.run(
            visual,
            steps=steps
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

    print()
    print(f"Direction {direction:+.2f}")
    print("Active receptors:", int(np.sum(visual > 0)))

    print(
        "Average spikes:",
        f"{np.mean(spike_values):.1f}"
    )

    print(
        "Steering:",
        f"L={steering_left/trials:.2f}",
        f"R={steering_right/trials:.2f}"
    )

    print(
        "Movement:",
        f"FL={forward_left/trials:.2f}",
        f"FR={forward_right/trials:.2f}",
        f"BL={backward_left/trials:.2f}",
        f"BR={backward_right/trials:.2f}"
    )

    print(
        "Escape:",
        f"L={escape_left/trials:.2f}",
        f"R={escape_right/trials:.2f}"
    )

    print(
        "Other:",
        f"PL={punch_left/trials:.2f}",
        f"PR={punch_right/trials:.2f}",
        f"KL={kick_left/trials:.2f}",
        f"KR={kick_right/trials:.2f}"
    )