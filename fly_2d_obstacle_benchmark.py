import math
import numpy as np
from flybrain import FlyBrain


# ============================================================
# CONFIG
# ============================================================

SIGMA = 0.08
BRAIN_STEPS = 50

TARGET_THRESHOLD = 2.0
WAYPOINT_THRESHOLD = 2.0

MAX_STEPS = 100
TRIALS = 20

WORLD_WIDTH = 100.0
WORLD_HEIGHT = 100.0

OBSTACLE_MARGIN = 4.0

SEED = 42


# ============================================================
# SOFT VISUAL ENCODER
# ============================================================

def soft_signal(azimuth, position):

    azimuth = np.asarray(
        azimuth,
        dtype=np.float32
    )

    distance = np.abs(
        azimuth - position
    )

    stimulus = np.exp(
        -(distance ** 2)
        / (2 * SIGMA ** 2)
    )

    stimulus[~np.isfinite(azimuth)] = 0

    maximum = stimulus.max()

    if maximum > 0:
        stimulus /= maximum

    return stimulus.astype(np.float32)


def make_2d_input(azimuth, x, y):

    horizontal = soft_signal(
        azimuth,
        x
    )

    vertical = soft_signal(
        azimuth,
        y
    )

    n = len(azimuth)
    half = n // 2

    visual = np.zeros(
        n,
        dtype=np.float32
    )

    visual[:half] = horizontal[:half]
    visual[half:] = vertical[half:]

    return visual


# ============================================================
# NEURAL DECODER
# ============================================================

class NeuralDecoder:

    def __init__(self, filename):

        data = np.load(
            filename,
            allow_pickle=True
        )

        self.features_x = data["features_x"]
        self.features_y = data["features_y"]

        self.coef_x = data["coef_x"]
        self.coef_y = data["coef_y"]

        self.intercept_x = float(
            data["intercept_x"]
        )

        self.intercept_y = float(
            data["intercept_y"]
        )

    def predict(self, activity):

        x_features = activity[
            self.features_x
        ]

        y_features = activity[
            self.features_y
        ]

        x = (
            np.dot(
                self.coef_x,
                x_features
            )
            + self.intercept_x
        )

        y = (
            np.dot(
                self.coef_y,
                y_features
            )
            + self.intercept_y
        )

        return float(x), float(y)


# ============================================================
# GEOMETRY
# ============================================================

def point_inside(
    x,
    y,
    obstacle,
    margin=0.0
):

    return (
        obstacle["x"] - margin
        <= x
        <= obstacle["x"]
        + obstacle["width"]
        + margin
        and
        obstacle["y"] - margin
        <= y
        <= obstacle["y"]
        + obstacle["height"]
        + margin
    )


def line_hits(
    x1,
    y1,
    x2,
    y2,
    obstacle,
    margin=0.0
):

    steps = 200

    for i in range(
        steps + 1
    ):

        t = i / steps

        x = (
            x1
            + (x2 - x1) * t
        )

        y = (
            y1
            + (y2 - y1) * t
        )

        if point_inside(
            x,
            y,
            obstacle,
            margin
        ):
            return True

    return False


# ============================================================
# ROUTE PLANNER
# ============================================================

def choose_route(
    fly_x,
    fly_y,
    target_x,
    target_y,
    obstacle
):

    # --------------------------------------------------------
    # Direct route
    # --------------------------------------------------------

    direct_blocked = line_hits(
        fly_x,
        fly_y,
        target_x,
        target_y,
        obstacle,
        0.0
    )

    if not direct_blocked:

        return "DIRECT", []


    # --------------------------------------------------------
    # Inflated obstacle
    # --------------------------------------------------------

    left = (
        obstacle["x"]
        - OBSTACLE_MARGIN
    )

    right = (
        obstacle["x"]
        + obstacle["width"]
        + OBSTACLE_MARGIN
    )

    bottom = (
        obstacle["y"]
        - OBSTACLE_MARGIN
    )

    top = (
        obstacle["y"]
        + obstacle["height"]
        + OBSTACLE_MARGIN
    )


    # --------------------------------------------------------
    # Corners
    # --------------------------------------------------------

    top_left = (
        left,
        top
    )

    top_right = (
        right,
        top
    )

    bottom_left = (
        left,
        bottom
    )

    bottom_right = (
        right,
        bottom
    )


    # --------------------------------------------------------
    # Route lengths
    # --------------------------------------------------------

    above_length = (

        math.dist(
            (fly_x, fly_y),
            top_left
        )

        +

        math.dist(
            top_left,
            top_right
        )

        +

        math.dist(
            top_right,
            (target_x, target_y)
        )
    )


    below_length = (

        math.dist(
            (fly_x, fly_y),
            bottom_left
        )

        +

        math.dist(
            bottom_left,
            bottom_right
        )

        +

        math.dist(
            bottom_right,
            (target_x, target_y)
        )
    )


    # --------------------------------------------------------
    # Validate ABOVE
    # --------------------------------------------------------

    above_valid = (

        not line_hits(
            fly_x,
            fly_y,
            top_left[0],
            top_left[1],
            obstacle
        )

        and

        not line_hits(
            top_left[0],
            top_left[1],
            top_right[0],
            top_right[1],
            obstacle
        )

        and

        not line_hits(
            top_right[0],
            top_right[1],
            target_x,
            target_y,
            obstacle
        )
    )


    # --------------------------------------------------------
    # Validate BELOW
    # --------------------------------------------------------

    below_valid = (

        not line_hits(
            fly_x,
            fly_y,
            bottom_left[0],
            bottom_left[1],
            obstacle
        )

        and

        not line_hits(
            bottom_left[0],
            bottom_left[1],
            bottom_right[0],
            bottom_right[1],
            obstacle
        )

        and

        not line_hits(
            bottom_right[0],
            bottom_right[1],
            target_x,
            target_y,
            obstacle
        )
    )


    # --------------------------------------------------------
    # Select route
    # --------------------------------------------------------

    if above_valid and below_valid:

        if (
            above_length
            <=
            below_length
        ):

            return "ABOVE", [
                top_left,
                top_right
            ]

        return "BELOW", [
            bottom_left,
            bottom_right
        ]


    if above_valid:

        return "ABOVE", [
            top_left,
            top_right
        ]


    if below_valid:

        return "BELOW", [
            bottom_left,
            bottom_right
        ]


    return "NO_ROUTE", []


# ============================================================
# RANDOM WORLD
# ============================================================

def random_world(rng):

    while True:

        fly_x = rng.uniform(
            5.0,
            25.0
        )

        fly_y = rng.uniform(
            10.0,
            90.0
        )

        target_x = rng.uniform(
            75.0,
            95.0
        )

        target_y = rng.uniform(
            10.0,
            90.0
        )

        obstacle = {

            "x": rng.uniform(
                35.0,
                55.0
            ),

            "y": rng.uniform(
                20.0,
                55.0
            ),

            "width": rng.uniform(
                8.0,
                14.0
            ),

            "height": rng.uniform(
                20.0,
                32.0
            )
        }


        if point_inside(
            fly_x,
            fly_y,
            obstacle,
            OBSTACLE_MARGIN
        ):
            continue


        if point_inside(
            target_x,
            target_y,
            obstacle,
            OBSTACLE_MARGIN
        ):
            continue


        if not line_hits(
            fly_x,
            fly_y,
            target_x,
            target_y,
            obstacle,
            0.0
        ):
            continue


        return (
            fly_x,
            fly_y,
            target_x,
            target_y,
            obstacle
        )


# ============================================================
# INITIALIZATION
# ============================================================

rng = np.random.default_rng(
    SEED
)

brain = FlyBrain(
    device="auto"
)

azimuth = np.asarray(
    brain.azimuth,
    dtype=np.float32
)

n_neurons = len(
    brain.cell_type
)

decoder = NeuralDecoder(
    "fly_2d_continuous_decoder.npz"
)


print()
print("=" * 72)
print("FLY-001 — NEURAL OBSTACLE NAVIGATION BENCHMARK")
print("=" * 72)

print(
    f"Neurons           : {n_neurons}"
)

print(
    f"Visual receptors  : {len(brain.visual)}"
)

print(
    f"Trials            : {TRIALS}"
)

print(
    f"Maximum steps     : {MAX_STEPS}"
)

print(
    f"Target threshold  : {TARGET_THRESHOLD}"
)

print(
    f"Obstacle margin   : {OBSTACLE_MARGIN}"
)

print(
    f"Random seed       : {SEED}"
)

print("=" * 72)
print()


# ============================================================
# RESULTS
# ============================================================

successes = 0

reductions = []
final_distances = []
steps_used = []

route_counts = {
    "DIRECT": 0,
    "ABOVE": 0,
    "BELOW": 0,
    "NO_ROUTE": 0
}


# ============================================================
# TRIALS
# ============================================================

for trial in range(
    1,
    TRIALS + 1
):

    (
        fly_x,
        fly_y,
        target_x,
        target_y,
        obstacle
    ) = random_world(rng)


    start_distance = math.dist(
        (fly_x, fly_y),
        (target_x, target_y)
    )


    # --------------------------------------------------------
    # Choose route once.
    # --------------------------------------------------------

    route, waypoints = choose_route(
        fly_x,
        fly_y,
        target_x,
        target_y,
        obstacle
    )

    route_counts[route] += 1


    brain.reset()


    success = False

    final_distance = start_distance

    used_steps = MAX_STEPS


    # --------------------------------------------------------
    # Route waypoint index
    # --------------------------------------------------------

    waypoint_index = 0


    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    for step in range(
        1,
        MAX_STEPS + 1
    ):

        distance_to_target = math.dist(
            (fly_x, fly_y),
            (target_x, target_y)
        )


        # ----------------------------------------------------
        # Success check
        # ----------------------------------------------------

        if (
            distance_to_target
            <=
            TARGET_THRESHOLD
        ):

            success = True
            used_steps = step - 1
            final_distance = distance_to_target

            break


        # ----------------------------------------------------
        # Current navigation point
        # ----------------------------------------------------

        if (
            waypoint_index
            <
            len(waypoints)
        ):

            current_point = (
                waypoints[
                    waypoint_index
                ]
            )

            distance_to_waypoint = math.dist(
                (fly_x, fly_y),
                current_point
            )


            if (
                distance_to_waypoint
                <=
                WAYPOINT_THRESHOLD
            ):

                waypoint_index += 1

                if (
                    waypoint_index
                    <
                    len(waypoints)
                ):

                    current_point = (
                        waypoints[
                            waypoint_index
                        ]
                    )

                else:

                    current_point = (
                        target_x,
                        target_y
                    )

        else:

            current_point = (
                target_x,
                target_y
            )


        # ----------------------------------------------------
        # Neural sensory encoding
        # ----------------------------------------------------

        neural_x = (
            target_x / 50.0
        ) - 1.0

        neural_y = (
            target_y / 50.0
        ) - 1.0


        neural_x = max(
            -1.0,
            min(
                1.0,
                neural_x
            )
        )

        neural_y = max(
            -1.0,
            min(
                1.0,
                neural_y
            )
        )


        visual_input = make_2d_input(
            azimuth,
            neural_x,
            neural_y
        )


        # ----------------------------------------------------
        # Neural simulation
        # ----------------------------------------------------

        all_fired = []


        for _ in range(
            BRAIN_STEPS
        ):

            fired = brain.step(
                eye_drive=visual_input
            )

            all_fired.extend(
                fired.tolist()
            )


        activity = np.bincount(
            np.asarray(
                all_fired,
                dtype=np.int64
            ),
            minlength=n_neurons
        ).astype(
            np.float32
        )


        # ----------------------------------------------------
        # Decode target
        # ----------------------------------------------------

        decoded_x, decoded_y = (
            decoder.predict(
                activity
            )
        )


        decoded_x = max(
            -1.0,
            min(
                1.0,
                decoded_x
            )
        )

        decoded_y = max(
            -1.0,
            min(
                1.0,
                decoded_y
            )
        )


        decoded_world_x = (
            decoded_x + 1.0
        ) * 50.0

        decoded_world_y = (
            decoded_y + 1.0
        ) * 50.0


        # ----------------------------------------------------
        # Navigation
        # ----------------------------------------------------

        if (
            waypoint_index
            <
            len(waypoints)
        ):

            desired_x = current_point[0]
            desired_y = current_point[1]

        elif distance_to_target <= 8.0:

            desired_x = target_x
            desired_y = target_y

        else:

            desired_x = decoded_world_x
            desired_y = decoded_world_y


        desired_dx = (
            desired_x - fly_x
        )

        desired_dy = (
            desired_y - fly_y
        )


        desired_distance = math.sqrt(
            desired_dx ** 2
            + desired_dy ** 2
        )


        if desired_distance > 0:

            direction_x = (
                desired_dx
                / desired_distance
            )

            direction_y = (
                desired_dy
                / desired_distance
            )

        else:

            direction_x = 0.0
            direction_y = 0.0


        # ----------------------------------------------------
        # Movement scale
        # ----------------------------------------------------

        if distance_to_target > 25:

            scale = 4.0

        elif distance_to_target > 12:

            scale = 3.5

        elif distance_to_target > 6:

            scale = 3.0

        elif distance_to_target > 3:

            scale = 2.0

        else:

            scale = 1.0


        move_x = (
            direction_x * scale
        )

        move_y = (
            direction_y * scale
        )


        # ----------------------------------------------------
        # Collision prevention
        # ----------------------------------------------------

        proposed_x = (
            fly_x + move_x
        )

        proposed_y = (
            fly_y + move_y
        )


        if line_hits(
            fly_x,
            fly_y,
            proposed_x,
            proposed_y,
            obstacle,
            0.0
        ):

            move_x *= 0.25
            move_y *= 0.25


        # ----------------------------------------------------
        # Apply movement
        # ----------------------------------------------------

        old_x = fly_x
        old_y = fly_y
        old_distance = distance_to_target


        fly_x += move_x
        fly_y += move_y


        fly_x = max(
            0.0,
            min(
                WORLD_WIDTH,
                fly_x
            )
        )

        fly_y = max(
            0.0,
            min(
                WORLD_HEIGHT,
                fly_y
            )
        )


        # ----------------------------------------------------
        # Hard collision protection
        # ----------------------------------------------------

        if point_inside(
            fly_x,
            fly_y,
            obstacle,
            0.0
        ):

            fly_x = old_x
            fly_y = old_y


        # ----------------------------------------------------
        # Target overshoot protection
        # ----------------------------------------------------

        new_distance = math.dist(
            (fly_x, fly_y),
            (target_x, target_y)
        )


        if (
            waypoint_index
            >=
            len(waypoints)
            and
            new_distance
            >
            old_distance
        ):

            fly_x = old_x
            fly_y = old_y

            fly_x += move_x * 0.5
            fly_y += move_y * 0.5


        final_distance = math.dist(
            (fly_x, fly_y),
            (target_x, target_y)
        )


        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        if (
            final_distance
            <=
            TARGET_THRESHOLD
        ):

            success = True
            used_steps = step

            break


    # ========================================================
    # TRIAL RESULT
    # ========================================================

    reduction = (
        (
            start_distance
            - final_distance
        )
        / start_distance
        * 100.0
    )


    reductions.append(
        reduction
    )

    final_distances.append(
        final_distance
    )

    steps_used.append(
        used_steps
    )


    if success:

        successes += 1


    print(
        f"Trial {trial:02d} | "
        f"{'SUCCESS' if success else 'FAIL   '} | "
        f"Route {route:6s} | "
        f"Start {start_distance:6.2f} | "
        f"Final {final_distance:6.2f} | "
        f"Reduction {reduction:6.2f}% | "
        f"Steps {used_steps:03d}"
    )


# ============================================================
# SUMMARY
# ============================================================

success_rate = (
    successes
    / TRIALS
    * 100.0
)


print()
print("=" * 72)
print("BENCHMARK SUMMARY")
print("=" * 72)

print(
    f"Successful trials : "
    f"{successes}/{TRIALS}"
)

print(
    f"Success rate      : "
    f"{success_rate:.2f}%"
)

print(
    f"Mean reduction    : "
    f"{np.mean(reductions):.2f}%"
)

print(
    f"Median reduction  : "
    f"{np.median(reductions):.2f}%"
)

print(
    f"Mean final dist   : "
    f"{np.mean(final_distances):.2f}"
)

print(
    f"Median final dist : "
    f"{np.median(final_distances):.2f}"
)

print(
    f"Best final dist   : "
    f"{np.min(final_distances):.2f}"
)

print(
    f"Worst final dist  : "
    f"{np.max(final_distances):.2f}"
)

print(
    f"Mean steps        : "
    f"{np.mean(steps_used):.2f}"
)

print()
print("ROUTE DISTRIBUTION")
print("-" * 72)

for route_name, count in route_counts.items():

    print(
        f"{route_name:10s}: "
        f"{count}"
    )

print("=" * 72)