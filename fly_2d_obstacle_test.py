import math
import numpy as np
from flybrain import FlyBrain


# ============================================================
# CONFIG
# ============================================================

SIGMA = 0.08
BRAIN_STEPS = 50

PREDICTION_HORIZON = 2.0

TARGET_THRESHOLD = 2.0

WORLD_WIDTH = 100.0
WORLD_HEIGHT = 100.0

MOVE_SCALE = 4.0

OBSTACLE = {
    "x": 45.0,
    "y": 35.0,
    "width": 10.0,
    "height": 30.0,
}

OBSTACLE_MARGIN = 3.0


# ============================================================
# WORLD
# ============================================================

class ObstacleWorld:

    def __init__(self):
        self.fly_x = 15.0
        self.fly_y = 50.0

        self.target_x = 85.0
        self.target_y = 50.0

        self.target_vx = 0.0
        self.target_vy = 0.0

        self.step_count = 0

    def distance_to_target(self):
        dx = self.target_x - self.fly_x
        dy = self.target_y - self.fly_y

        return math.sqrt(dx * dx + dy * dy)

    def move_fly(self, dx, dy):

        new_x = self.fly_x + dx
        new_y = self.fly_y + dy

        new_x = max(0.0, min(WORLD_WIDTH, new_x))
        new_y = max(0.0, min(WORLD_HEIGHT, new_y))

        self.fly_x = new_x
        self.fly_y = new_y

    def target_direction(self):

        dx = self.target_x - self.fly_x
        dy = self.target_y - self.fly_y

        distance = math.sqrt(dx * dx + dy * dy)

        if distance == 0:
            return 0.0, 0.0, 0.0

        return dx, dy, distance

    def describe(self):

        print()
        print("=" * 60)
        print("OBSTACLE NAVIGATION TEST")
        print("=" * 60)

        print(
            f"Fly       : ({self.fly_x:.2f}, {self.fly_y:.2f})"
        )

        print(
            f"Target    : ({self.target_x:.2f}, {self.target_y:.2f})"
        )

        print(
            f"Distance  : {self.distance_to_target():.2f}"
        )

        print(
            f"Obstacle  : "
            f"x={OBSTACLE['x']}, "
            f"y={OBSTACLE['y']}, "
            f"w={OBSTACLE['width']}, "
            f"h={OBSTACLE['height']}"
        )

        print("=" * 60)


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

    visual = np.zeros(
        n,
        dtype=np.float32
    )

    half = n // 2

    visual[:half] = horizontal[:half]
    visual[half:] = vertical[half:]

    return visual


# ============================================================
# DECODER
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
# OBSTACLE GEOMETRY
# ============================================================

def point_inside_obstacle(
    x,
    y,
    margin=0.0
):

    left = OBSTACLE["x"] - margin

    right = (
        OBSTACLE["x"]
        + OBSTACLE["width"]
        + margin
    )

    bottom = OBSTACLE["y"] - margin

    top = (
        OBSTACLE["y"]
        + OBSTACLE["height"]
        + margin
    )

    return (
        left <= x <= right
        and
        bottom <= y <= top
    )


def line_hits_obstacle(
    x1,
    y1,
    x2,
    y2,
    margin=0.0,
    samples=25
):

    for i in range(samples + 1):

        t = i / samples

        x = x1 + (
            x2 - x1
        ) * t

        y = y1 + (
            y2 - y1
        ) * t

        if point_inside_obstacle(
            x,
            y,
            margin
        ):
            return True

    return False


# ============================================================
# OBSTACLE AVOIDANCE
# ============================================================

def avoid_obstacle(
    fly_x,
    fly_y,
    desired_dx,
    desired_dy
):

    target_x = fly_x + desired_dx
    target_y = fly_y + desired_dy

    # Direct route is clear.
    if not line_hits_obstacle(
        fly_x,
        fly_y,
        target_x,
        target_y,
        OBSTACLE_MARGIN
    ):

        return (
            desired_dx,
            desired_dy,
            False
        )

    # --------------------------------------------------------
    # Try ABOVE obstacle.
    # --------------------------------------------------------

    top_y = (
        OBSTACLE["y"]
        + OBSTACLE["height"]
        + OBSTACLE_MARGIN
        + 1.0
    )

    dx_top = target_x - fly_x
    dy_top = top_y - fly_y

    top_blocked = line_hits_obstacle(
        fly_x,
        fly_y,
        fly_x + dx_top,
        fly_y + dy_top,
        OBSTACLE_MARGIN
    )

    # --------------------------------------------------------
    # Try BELOW obstacle.
    # --------------------------------------------------------

    bottom_y = (
        OBSTACLE["y"]
        - OBSTACLE_MARGIN
        - 1.0
    )

    dx_bottom = target_x - fly_x
    dy_bottom = bottom_y - fly_y

    bottom_blocked = line_hits_obstacle(
        fly_x,
        fly_y,
        fly_x + dx_bottom,
        fly_y + dy_bottom,
        OBSTACLE_MARGIN
    )

    # --------------------------------------------------------
    # Choose route.
    # --------------------------------------------------------

    if not top_blocked and not bottom_blocked:

        top_distance = math.sqrt(
            dx_top ** 2
            + dy_top ** 2
        )

        bottom_distance = math.sqrt(
            dx_bottom ** 2
            + dy_bottom ** 2
        )

        if top_distance <= bottom_distance:

            return (
                dx_top,
                dy_top,
                True
            )

        return (
            dx_bottom,
            dy_bottom,
            True
        )

    if not top_blocked:

        return (
            dx_top,
            dy_top,
            True
        )

    if not bottom_blocked:

        return (
            dx_bottom,
            dy_bottom,
            True
        )

    # Emergency vertical escape.

    obstacle_center_y = (
        OBSTACLE["y"]
        + OBSTACLE["height"] / 2
    )

    if fly_y < obstacle_center_y:

        return (
            desired_dx * 0.2,
            abs(desired_dy) + MOVE_SCALE,
            True
        )

    return (
        desired_dx * 0.2,
        -abs(desired_dy) - MOVE_SCALE,
        True
    )


# ============================================================
# INITIALIZE BRAIN
# ============================================================

brain = FlyBrain(
    device="auto"
)

azimuth = np.asarray(
    brain.azimuth,
    dtype=np.float32
)

n_visual = len(
    brain.visual
)

print()
print("FLY BRAIN")
print("-" * 60)

print(
    f"Neurons          : "
    f"{len(brain.cell_type)}"
)

print(
    f"Visual receptors : "
    f"{n_visual}"
)

print(
    f"Prediction horizon: "
    f"{PREDICTION_HORIZON}"
)

print("-" * 60)


# ============================================================
# LOAD DECODER
# ============================================================

decoder = NeuralDecoder(
    "fly_2d_continuous_decoder.npz"
)

print("Decoder loaded.")


# ============================================================
# CREATE WORLD
# ============================================================

world = ObstacleWorld()

world.describe()


# ============================================================
# MAIN LOOP
# ============================================================

MAX_STEPS = 60

print()
print("Starting obstacle navigation...")
print()


for step in range(
    1,
    MAX_STEPS + 1
):

    distance_before = (
        world.distance_to_target()
    )

    # --------------------------------------------------------
    # Predict target position.
    # Target is stationary in this test.
    # --------------------------------------------------------

    predicted_x = (
        world.target_x
        + world.target_vx
        * PREDICTION_HORIZON
    )

    predicted_y = (
        world.target_y
        + world.target_vy
        * PREDICTION_HORIZON
    )

    predicted_x = max(
        0.0,
        min(
            WORLD_WIDTH,
            predicted_x
        )
    )

    predicted_y = max(
        0.0,
        min(
            WORLD_HEIGHT,
            predicted_y
        )
    )

    # --------------------------------------------------------
    # Convert world coordinates to
    # neural coordinates [-1, 1].
    # --------------------------------------------------------

    neural_x = (
        predicted_x / 50.0
    ) - 1.0

    neural_y = (
        predicted_y / 50.0
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

    # --------------------------------------------------------
    # Encode visual input.
    # --------------------------------------------------------

    visual_input = make_2d_input(
        azimuth,
        neural_x,
        neural_y
    )

    # --------------------------------------------------------
    # Run FlyBrain.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Neural population activity.
    # --------------------------------------------------------

    activity = np.bincount(
        np.asarray(
            all_fired,
            dtype=np.int64
        ),
        minlength=len(
            brain.cell_type
        )
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Decode.
    # --------------------------------------------------------

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

    # ========================================================
    # HYBRID TARGET APPROACH
    # ========================================================
    #
    # Far away:
    #     use neural decoder.
    #
    # Close:
    #     use precise target direction.
    #
    # This prevents decoder noise from causing
    # oscillation near the target.
    # ========================================================

    if distance_before <= 8.0:

        desired_dx = (
            world.target_x
            - world.fly_x
        )

        desired_dy = (
            world.target_y
            - world.fly_y
        )

        control_mode = "TARGET"

    else:

        desired_dx = (
            decoded_world_x
            - world.fly_x
        )

        desired_dy = (
            decoded_world_y
            - world.fly_y
        )

        control_mode = "NEURAL"

    desired_distance = math.sqrt(
        desired_dx ** 2
        + desired_dy ** 2
    )

    # --------------------------------------------------------
    # Normalize direction.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Adaptive movement.
    # --------------------------------------------------------

    if distance_before > 20:

        scale = 4.0

    elif distance_before > 10:

        scale = 3.0

    elif distance_before > 5:

        scale = 2.0

    else:

        scale = 1.0

    move_x = (
        direction_x
        * scale
    )

    move_y = (
        direction_y
        * scale
    )

    # --------------------------------------------------------
    # Obstacle avoidance.
    # --------------------------------------------------------

    move_x, move_y, avoiding = (
        avoid_obstacle(
            world.fly_x,
            world.fly_y,
            move_x,
            move_y
        )
    )

    # --------------------------------------------------------
    # Keep movement within allowed scale.
    # --------------------------------------------------------

    move_distance = math.sqrt(
        move_x ** 2
        + move_y ** 2
    )

    if move_distance > scale:

        move_x = (
            move_x
            / move_distance
            * scale
        )

        move_y = (
            move_y
            / move_distance
            * scale
        )

    # --------------------------------------------------------
    # Overshoot protection.
    # --------------------------------------------------------

    old_x = world.fly_x
    old_y = world.fly_y

    world.move_fly(
        move_x,
        move_y
    )

    new_distance = (
        world.distance_to_target()
    )

    if new_distance > distance_before:

        world.fly_x = old_x
        world.fly_y = old_y

        move_x *= 0.5
        move_y *= 0.5

        world.move_fly(
            move_x,
            move_y
        )

    world.step_count += 1

    final_distance = (
        world.distance_to_target()
    )

    # --------------------------------------------------------
    # Output.
    # --------------------------------------------------------

    print(
        f"Step {step:02d} | "
        f"Dist {distance_before:6.2f} -> "
        f"{final_distance:6.2f} | "
        f"Decoded "
        f"({decoded_world_x:6.2f}, "
        f"{decoded_world_y:6.2f}) | "
        f"Move "
        f"({move_x:5.2f}, "
        f"{move_y:5.2f}) | "
        f"{'AVOID' if avoiding else control_mode}"
    )

    # --------------------------------------------------------
    # Success.
    # --------------------------------------------------------

    if final_distance <= TARGET_THRESHOLD:

        print()
        print("=" * 60)
        print("TARGET REACHED")
        print("=" * 60)

        print(
            f"Steps          : "
            f"{step}"
        )

        print(
            f"Final distance : "
            f"{final_distance:.2f}"
        )

        print(
            f"Final fly      : "
            f"({world.fly_x:.2f}, "
            f"{world.fly_y:.2f})"
        )

        print(
            f"Target         : "
            f"({world.target_x:.2f}, "
            f"{world.target_y:.2f})"
        )

        print("=" * 60)

        break

else:

    print()
    print("=" * 60)
    print("TARGET NOT REACHED")
    print("=" * 60)

    print(
        f"Final distance : "
        f"{world.distance_to_target():.2f}"
    )

    print(
        f"Final fly      : "
        f"({world.fly_x:.2f}, "
        f"{world.fly_y:.2f})"
    )

    print("=" * 60)