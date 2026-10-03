import numpy as np
from fly_core import Fly001
from fly_vision import FlyVision


fly = Fly001()
vision = FlyVision(fly)

direction = 0.0
target_receptors = 500

on_steps = 50
off_steps = 50
trials = 20


def make_stimulus():

    distance = np.abs(
        vision.azimuth - direction
    )

    order = np.argsort(distance)

    selected = order[:target_receptors]

    visual = np.zeros(
        vision.n,
        dtype=np.float32
    )

    visual[selected] = 1.0

    return visual


light = make_stimulus()

dark = np.zeros(
    vision.n,
    dtype=np.float32
)


print("FLY-001 ON/OFF response experiment")
print("Direction:", direction)
print("Active receptors:", target_receptors)
print("ON duration:", on_steps * 20, "ms")
print("OFF duration:", off_steps * 20, "ms")
print("Trials:", trials)


all_results = []


for trial in range(trials):

    fly.reset()

    trial_spikes = []

    # LIGHT ON
    for step in range(on_steps):

        fired = fly.step(light)

        trial_spikes.append(
            len(fired)
        )

    # LIGHT OFF
    for step in range(off_steps):

        fired = fly.step(dark)

        trial_spikes.append(
            len(fired)
        )

    all_results.append(trial_spikes)


average = np.mean(
    all_results,
    axis=0
)

std = np.std(
    all_results,
    axis=0
)


print("\n========== RESULTS ==========")

print(
    f"{'Step':>5} "
    f"{'Time(ms)':>8} "
    f"{'State':>8} "
    f"{'Avg spikes':>12} "
    f"{'Std':>10}"
)

print("-" * 50)


for step in range(on_steps + off_steps):

    time_ms = step * 20

    if step < on_steps:
        state = "ON"
    else:
        state = "OFF"

    print(
        f"{step:5d} "
        f"{time_ms:8d} "
        f"{state:>8} "
        f"{average[step]:12.1f} "
        f"{std[step]:10.1f}"
    )


print("\n========== SUMMARY ==========")

on_final = np.mean(
    average[on_steps - 5:on_steps]
)

off_first = np.mean(
    average[on_steps:on_steps + 5]
)

off_final = np.mean(
    average[-5:]
)

print(
    f"Final 100 ms ON:   {on_final:.1f}"
)

print(
    f"First 100 ms OFF:  {off_first:.1f}"
)

print(
    f"Final 100 ms OFF:  {off_final:.1f}"
)

print(
    f"OFF reduction:     "
    f"{((on_final - off_final) / on_final) * 100:.2f}%"
)