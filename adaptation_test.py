import numpy as np
from fly_core import Fly001
from fly_vision import FlyVision


fly = Fly001()
vision = FlyVision(fly)

direction = 0.0
target_receptors = 500
intensity = 1.0

steps = 100
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

    visual[selected] = intensity

    return visual


visual = make_stimulus()


print("FLY-001 adaptation / habituation experiment")
print("Direction:", direction)
print("Target receptors:", target_receptors)
print("Intensity:", intensity)
print("Steps:", steps)
print("Trials:", trials)

print("\nActive receptors:", int(np.sum(visual > 0)))


all_results = []


for trial in range(trials):

    fly.reset()

    trial_spikes = []

    for step in range(steps):

        fired = fly.step(visual)

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
    f"{'Avg spikes':>12} "
    f"{'Std':>10}"
)

print("-" * 42)


for step in range(steps):

    time_ms = step * 20

    print(
        f"{step:5d} "
        f"{time_ms:8d} "
        f"{average[step]:12.1f} "
        f"{std[step]:10.1f}"
    )


print("\n========== SUMMARY ==========")

first_5 = np.mean(
    average[:5]
)

last_5 = np.mean(
    average[-5:]
)

peak = np.max(average)

peak_step = np.argmax(average)

print(
    f"First 5 steps average: {first_5:.1f}"
)

print(
    f"Last 5 steps average:  {last_5:.1f}"
)

print(
    f"Peak response:          {peak:.1f}"
)

print(
    f"Peak time:              {peak_step * 20} ms"
)

print(
    f"Change first→last:      "
    f"{((last_5 - first_5) / first_5) * 100:.2f}%"
)