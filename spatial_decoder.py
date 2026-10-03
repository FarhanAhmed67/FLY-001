import numpy as np
from flybrain import FlyBrain
from fly_core import Fly001
from fly_vision import FlyVision


fly = Fly001()
vision = FlyVision(fly)
brain = fly.brain

target_receptors = 500

trials = 20
stage_steps = 50


directions = {
    "LEFT": -0.75,
    "CENTER_LEFT": -0.375,
    "CENTER": 0.0,
    "CENTER_RIGHT": 0.375,
    "RIGHT": 0.75
}


def make_stimulus(direction):

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


stimuli = {
    name: make_stimulus(direction)
    for name, direction in directions.items()
}


def record_state(visual):

    counts = np.zeros(
        len(brain.cell_type),
        dtype=np.float32
    )

    total_spikes = 0

    for _ in range(stage_steps):

        fired = fly.step(
            visual
        )

        fired = np.asarray(
            fired,
            dtype=np.int64
        )

        total_spikes += len(fired)

        if len(fired) > 0:

            unique, frequency = np.unique(
                fired,
                return_counts=True
            )

            counts[unique] += frequency

    return {
        "rates": counts / stage_steps,
        "spikes": total_spikes / stage_steps
    }


results = {
    direction: []
    for direction in directions
}


print(
    "FLY-001 spatial decoder experiment"
)

print(
    "500 receptors per stimulus"
)

print(
    "Positions:",
    ", ".join(directions.keys())
)

print(
    "Trials:",
    trials
)

print(
    "Measurement:",
    stage_steps,
    "steps = 1000 ms"
)


for direction_name in directions:

    print(
        f"\n========== {direction_name} =========="
    )

    stimulus = stimuli[
        direction_name
    ]

    for trial in range(trials):

        print(
            f"{direction_name}: "
            f"trial {trial + 1}/{trials}"
        )

        fly.reset()

        state = record_state(
            stimulus
        )

        results[
            direction_name
        ].append(state)


print(
    "\n\n========== SPIKE RATE =========="
)


for direction_name in directions:

    values = [
        result["spikes"]
        for result in results[
            direction_name
        ]
    ]

    print(
        f"{direction_name:14} "
        f"{np.mean(values):.2f}"
    )


print(
    "\n\n========== PAIRWISE NEURAL SIMILARITY =========="
)


def cosine_similarity(a, b):

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return np.dot(a, b) / denominator


direction_names = list(
    directions.keys()
)


mean_states = {}

for direction_name in direction_names:

    mean_states[
        direction_name
    ] = np.mean(
        [
            result["rates"]
            for result in results[
                direction_name
            ]
        ],
        axis=0
    )


print(
    "\nCosine similarity:"
)

print(
    "Rows = first position"
)

print(
    "Columns = second position\n"
)


print(
    "               "
    + " ".join(
        f"{name[:7]:>9}"
        for name in direction_names
    )
)


for name_a in direction_names:

    row = []

    for name_b in direction_names:

        similarity = cosine_similarity(
            mean_states[name_a],
            mean_states[name_b]
        )

        row.append(
            f"{similarity:9.4f}"
        )

    print(
        f"{name_a[:14]:14}"
        + "".join(row)
    )


print(
    "\n\n========== TOP DIRECTION-SENSITIVE NEURONS =========="
)


all_states = np.array([
    mean_states[name]
    for name in direction_names
])


neuron_variance = np.var(
    all_states,
    axis=0
)

ranking = np.argsort(
    neuron_variance
)[::-1]


print(
    "\nRank | Neuron | "
    "LEFT | C-L | CENTER | C-R | RIGHT"
)


for rank, neuron in enumerate(
    ranking[:50],
    start=1
):

    values = [
        mean_states[name][neuron]
        for name in direction_names
    ]

    print(
        f"{rank:4d} | "
        f"{neuron:6d} | "
        f"{values[0]:5.2f} | "
        f"{values[1]:5.2f} | "
        f"{values[2]:6.2f} | "
        f"{values[3]:5.2f} | "
        f"{values[4]:5.2f}"
    )


print(
    "\n\n========== POSITION RESPONSE PATTERNS =========="
)


for top_n in [
    10,
    50,
    100,
    500,
    1000,
    5000
]:

    selected = ranking[:top_n]

    print(
        f"\nTOP {top_n}"
    )

    for name in direction_names:

        values = []

        for trial in results[name]:

            values.append(
                np.mean(
                    trial["rates"][selected]
                )
            )

        print(
            f"  {name:14} "
            f"{np.mean(values):.4f}"
        )


print(
    "\nExperiment complete."
)