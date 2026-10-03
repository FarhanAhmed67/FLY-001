import numpy as np
from flybrain import FlyBrain

WORLD_WIDTH = 100.0
WORLD_HEIGHT = 100.0

SIGMA = 0.08
BRAIN_STEPS = 50

TRIALS = 20
MAX_STEPS = 100

TARGET_SPEED = 3.0
PREDICTION_HORIZON = 2.0
TARGET_THRESHOLD = 2.0

NEURAL_UPDATE_EVERY = 5
OBSTACLE_MARGIN = 4.0
RANDOM_SEED = 42

MODEL_FILE = "fly_2d_continuous_decoder.npz"
N_NEURONS = 166700


def soft_signal(azimuth, position):
    distance = np.abs(azimuth - position)

    signal = np.exp(
        -(distance ** 2) / (2.0 * SIGMA ** 2)
    )

    signal[~np.isfinite(azimuth)] = 0.0

    maximum = signal.max()

    if maximum > 0:
        signal /= maximum

    return signal.astype(np.float32)


def encode_2d(azimuth, x_position, y_position):
    signal_x = soft_signal(azimuth, x_position)
    signal_y = soft_signal(azimuth, y_position)

    midpoint = len(azimuth) // 2

    stimulus = np.zeros(
        len(azimuth),
        dtype=np.float32
    )

    stimulus[:midpoint] = signal_x[:midpoint]
    stimulus[midpoint:] = signal_y[midpoint:]

    return stimulus


class NeuralDecoder:
    def __init__(self, filename):
        self.model = np.load(
            filename,
            allow_pickle=False
        )

        self.features_x = self.model["features_x"]
        self.features_y = self.model["features_y"]

        self.coef_x = self.model["coef_x"]
        self.coef_y = self.model["coef_y"]

        self.intercept_x = float(
            self.model["intercept_x"]
        )

        self.intercept_y = float(
            self.model["intercept_y"]
        )

    def decode(self, activity):
        pred_x = float(
            activity[self.features_x] @ self.coef_x
            + self.intercept_x
        )

        pred_y = float(
            activity[self.features_y] @ self.coef_y
            + self.intercept_y
        )

        return pred_x, pred_y


def world_to_decoder(x, y):
    decoder_x = (
        (x / WORLD_WIDTH) * 2.0
        - 1.0
    )

    decoder_y = (
        (y / WORLD_HEIGHT) * 2.0
        - 1.0
    )

    return decoder_x, decoder_y


def decoder_to_world(x, y):
    world_x = (
        (x + 1.0) / 2.0 * WORLD_WIDTH
    )

    world_y = (
        (y + 1.0) / 2.0 * WORLD_HEIGHT
    )

    return world_x, world_y


def neural_localize(
    brain,
    azimuth,
    decoder,
    world_x,
    world_y
):
    decoder_x, decoder_y = world_to_decoder(
        world_x,
        world_y
    )

    stimulus = encode_2d(
        azimuth,
        decoder_x,
        decoder_y
    )

    brain.reset()

    activity = np.zeros(
        N_NEURONS,
        dtype=np.float32
    )

    for _ in range(BRAIN_STEPS):
        fired = brain.step(
            eye_drive=stimulus
        )

        if len(fired) > 0:
            activity[fired] += 1.0

    predicted_x, predicted_y = decoder.decode(
        activity
    )

    predicted_world_x, predicted_world_y = decoder_to_world(
        predicted_x,
        predicted_y
    )

    predicted_world_x = np.clip(
        predicted_world_x,
        0.0,
        WORLD_WIDTH
    )

    predicted_world_y = np.clip(
        predicted_world_y,
        0.0,
        WORLD_HEIGHT
    )

    return predicted_world_x, predicted_world_y


def point_in_rect(point, rect):
    x, y = point
    rx, ry, rw, rh = rect

    return (
        rx <= x <= rx + rw
        and
        ry <= y <= ry + rh
    )


def segment_intersects_rect(p1, p2, rect):
    x1, y1 = p1
    x2, y2 = p2

    rx, ry, rw, rh = rect

    left = rx
    right = rx + rw
    bottom = ry
    top = ry + rh

    dx = x2 - x1
    dy = y2 - y1

    if (
        left <= x1 <= right
        and
        bottom <= y1 <= top
    ):
        return True

    if (
        left <= x2 <= right
        and
        bottom <= y2 <= top
    ):
        return True

    if abs(dx) < 1e-12:
        if x1 < left or x1 > right:
            return False

        ymin = min(y1, y2)
        ymax = max(y1, y2)

        return not (
            ymax < bottom
            or
            ymin > top
        )

    if abs(dy) < 1e-12:
        if y1 < bottom or y1 > top:
            return False

        xmin = min(x1, x2)
        xmax = max(x1, x2)

        return not (
            xmax < left
            or
            xmin > right
        )

    p = [
        -dx,
        dx,
        -dy,
        dy
    ]

    q = [
        x1 - left,
        right - x1,
        y1 - bottom,
        top - y1
    ]

    t_min = 0.0
    t_max = 1.0

    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return False
        else:
            t = qi / pi

            if pi < 0:
                if t > t_max:
                    return False

                if t > t_min:
                    t_min = t

            else:
                if t < t_min:
                    return False

                if t < t_max:
                    t_max = t

    return t_min <= t_max


def segment_clear(p1, p2, obstacle):
    return not segment_intersects_rect(
        p1,
        p2,
        obstacle
    )


class MovingWorld:

    def __init__(self, rng):

        self.obstacle = (
            45.0,
            35.0,
            10.0,
            30.0
        )

        self.fly_x = rng.uniform(
            5.0,
            25.0
        )

        self.fly_y = rng.uniform(
            10.0,
            90.0
        )

        self.target_x = rng.uniform(
            75.0,
            95.0
        )

        self.target_y = rng.uniform(
            10.0,
            90.0
        )

        angle = rng.uniform(
            0.0,
            2.0 * np.pi
        )

        self.target_vx = (
            TARGET_SPEED
            * np.cos(angle)
        )

        self.target_vy = (
            TARGET_SPEED
            * np.sin(angle)
        )

    def update_target(self):

        old_position = (
            self.target_x,
            self.target_y
        )

        proposed_position = (
            self.target_x + self.target_vx,
            self.target_y + self.target_vy
        )

        if segment_intersects_rect(
            old_position,
            proposed_position,
            self.obstacle
        ):

            x_only = (
                self.target_x + self.target_vx,
                self.target_y
            )

            y_only = (
                self.target_x,
                self.target_y + self.target_vy
            )

            hit_x = segment_intersects_rect(
                old_position,
                x_only,
                self.obstacle
            )

            hit_y = segment_intersects_rect(
                old_position,
                y_only,
                self.obstacle
            )

            if hit_x:
                self.target_vx = -self.target_vx

            if hit_y:
                self.target_vy = -self.target_vy

            if not hit_x and not hit_y:
                self.target_vx = -self.target_vx
                self.target_vy = -self.target_vy

        self.target_x += self.target_vx
        self.target_y += self.target_vy

        if self.target_x <= 5.0:

            self.target_x = 5.0

            self.target_vx = abs(
                self.target_vx
            )

        elif self.target_x >= 95.0:

            self.target_x = 95.0

            self.target_vx = -abs(
                self.target_vx
            )

        if self.target_y <= 5.0:

            self.target_y = 5.0

            self.target_vy = abs(
                self.target_vy
            )

        elif self.target_y >= 95.0:

            self.target_y = 95.0

            self.target_vy = -abs(
                self.target_vy
            )


def plan_route(start, target, obstacle):
    rx, ry, rw, rh = obstacle

    margin = OBSTACLE_MARGIN
    epsilon = 0.01

    # Expanded obstacle boundary.
    left_x = (
        rx - margin - epsilon
    )

    right_x = (
        rx + rw + margin + epsilon
    )

    bottom_y = (
        ry - margin - epsilon
    )

    top_y = (
        ry + rh + margin + epsilon
    )

    # Four navigational corners.
    top_left = (
        left_x,
        top_y
    )

    top_right = (
        right_x,
        top_y
    )

    bottom_left = (
        left_x,
        bottom_y
    )

    bottom_right = (
        right_x,
        bottom_y
    )

    corners = [
        ("TL", top_left),
        ("TR", top_right),
        ("BL", bottom_left),
        ("BR", bottom_right)
    ]

    # Direct path first.
    if segment_clear(
        start,
        target,
        obstacle
    ):
        return "DIRECT", [target]

    candidates = []

    # ---------------------------------------------------------
    # One-corner routes
    # ---------------------------------------------------------

    for name, corner in corners:

        if (
            segment_clear(
                start,
                corner,
                obstacle
            )
            and
            segment_clear(
                corner,
                target,
                obstacle
            )
        ):

            distance = (
                np.sqrt(
                    (
                        corner[0]
                        - start[0]
                    ) ** 2
                    +
                    (
                        corner[1]
                        - start[1]
                    ) ** 2
                )
                +
                np.sqrt(
                    (
                        target[0]
                        - corner[0]
                    ) ** 2
                    +
                    (
                        target[1]
                        - corner[1]
                    ) ** 2
                )
            )

            candidates.append(
                (
                    distance,
                    name,
                    [corner, target]
                )
            )

    # ---------------------------------------------------------
    # Two-corner routes
    # ---------------------------------------------------------

    for name_a, corner_a in corners:

        for name_b, corner_b in corners:

            if name_a == name_b:
                continue

            if not segment_clear(
                start,
                corner_a,
                obstacle
            ):
                continue

            if not segment_clear(
                corner_a,
                corner_b,
                obstacle
            ):
                continue

            if not segment_clear(
                corner_b,
                target,
                obstacle
            ):
                continue

            distance = (
                np.sqrt(
                    (
                        corner_a[0]
                        - start[0]
                    ) ** 2
                    +
                    (
                        corner_a[1]
                        - start[1]
                    ) ** 2
                )
                +
                np.sqrt(
                    (
                        corner_b[0]
                        - corner_a[0]
                    ) ** 2
                    +
                    (
                        corner_b[1]
                        - corner_a[1]
                    ) ** 2
                )
                +
                np.sqrt(
                    (
                        target[0]
                        - corner_b[0]
                    ) ** 2
                    +
                    (
                        target[1]
                        - corner_b[1]
                    ) ** 2
                )
            )

            candidates.append(
                (
                    distance,
                    name_a + "-" + name_b,
                    [corner_a, corner_b, target]
                )
            )

    # ---------------------------------------------------------
    # Select shortest valid route.
    # ---------------------------------------------------------

    if not candidates:
        return "NO_ROUTE", []

    candidates.sort(
        key=lambda item: item[0]
    )

    _, route_code, waypoints = candidates[0]

    if "TL" in route_code or "TR" in route_code:
        route_name = "ABOVE"
    else:
        route_name = "BELOW"

    return route_name, waypoints

def route_planner_test():
    print()
    print("Route planner check:")

    obstacle = (
        45.0,
        35.0,
        10.0,
        30.0
    )

    start = (
        15.0,
        50.0
    )

    target = (
        85.0,
        50.0
    )

    route, waypoints = plan_route(
        start,
        target,
        obstacle
    )

    print(
        "  Expected : ABOVE or BELOW"
    )

    print(
        f"  Actual   : {route}"
    )

    print(
        "  Waypoints:"
    )

    for waypoint in waypoints:
        print(
            f"    ({waypoint[0]:.2f}, "
            f"{waypoint[1]:.2f})"
        )


def move_toward(
    position,
    destination,
    obstacle,
    step_size
):
    x, y = position
    tx, ty = destination

    dx = tx - x
    dy = ty - y

    distance = np.sqrt(
        dx * dx + dy * dy
    )

    if distance < 1e-9:
        return x, y

    scale = min(
        step_size / distance,
        1.0
    )

    new_x = x + dx * scale
    new_y = y + dy * scale

    new_position = (
        new_x,
        new_y
    )

    if segment_intersects_rect(
        position,
        new_position,
        obstacle
    ):
        return x, y

    new_x = np.clip(
        new_x,
        0.0,
        WORLD_WIDTH
    )

    new_y = np.clip(
        new_y,
        0.0,
        WORLD_HEIGHT
    )

    return new_x, new_y


def adaptive_step(
    position,
    destination,
    obstacle
):
    distance = np.sqrt(
        (
            destination[0]
            - position[0]
        ) ** 2
        +
        (
            destination[1]
            - position[1]
        ) ** 2
    )

    if distance > 30.0:
        step_size = 4.0
    elif distance > 10.0:
        step_size = 3.0
    else:
        step_size = 2.0

    return move_toward(
        position,
        destination,
        obstacle,
        step_size
    )


def run_trial(
    brain,
    azimuth,
    decoder,
    rng,
    trial_index=None
):
    world = MovingWorld(rng)

    current_position = (
        world.fly_x,
        world.fly_y
    )

    start_target = (
        world.target_x,
        world.target_y
    )

    start_distance = np.sqrt(
        (
            start_target[0]
            - current_position[0]
        ) ** 2
        +
        (
            start_target[1]
            - current_position[1]
        ) ** 2
    )

    route_name = "NO_ROUTE"
    waypoints = []
    waypoint_index = 0

    direct_blocked = False
    route_history = []

    previous_target_x = world.target_x
    previous_target_y = world.target_y

    for step in range(MAX_STEPS):

        world.update_target()

        actual_target = (
            world.target_x,
            world.target_y
        )

        measured_vx = (
            world.target_x
            - previous_target_x
        )

        measured_vy = (
            world.target_y
            - previous_target_y
        )

        previous_target_x = world.target_x
        previous_target_y = world.target_y

        if step % NEURAL_UPDATE_EVERY == 0:

            neural_x, neural_y = neural_localize(
                brain,
                azimuth,
                decoder,
                current_position[0],
                current_position[1]
            )

            predicted_target_x = (
                world.target_x
                + measured_vx
                * PREDICTION_HORIZON
            )

            predicted_target_y = (
                world.target_y
                + measured_vy
                * PREDICTION_HORIZON
            )

            predicted_target = (
                np.clip(
                    predicted_target_x,
                    0.0,
                    WORLD_WIDTH
                ),
                np.clip(
                    predicted_target_y,
                    0.0,
                    WORLD_HEIGHT
                )
            )

            # A short-horizon prediction can overshoot into the
            # obstacle even though the actual target is outside it.
            # Never use an impossible point inside the obstacle
            # as the navigation destination.
            obstacle_x, obstacle_y, obstacle_w, obstacle_h = (
                world.obstacle
            )

            predicted_inside_obstacle = (
                obstacle_x
                <= predicted_target[0]
                <= obstacle_x + obstacle_w
                and
                obstacle_y
                <= predicted_target[1]
                <= obstacle_y + obstacle_h
            )

            if predicted_inside_obstacle:
                predicted_target = actual_target

            actual_direct_blocked = segment_intersects_rect(
                current_position,
                predicted_target,
                world.obstacle
            )

            neural_direct_blocked = segment_intersects_rect(
                (
                    neural_x,
                    neural_y
                ),
                predicted_target,
                world.obstacle
            )

            if actual_direct_blocked:
                direct_blocked = True

            if actual_direct_blocked:
                route_name, waypoints = plan_route(
                    current_position,
                    predicted_target,
                    world.obstacle
                )
            else:
                route_name = "DIRECT"
                waypoints = [
                    predicted_target
                ]

            route_history.append(
                (
                    actual_direct_blocked,
                    neural_direct_blocked,
                    route_name
                )
            )

            if (
                actual_direct_blocked
                and
                route_name == "NO_ROUTE"
                and
                len([
                    r
                    for r in route_history
                    if r[2] == "NO_ROUTE"
                ]) <= 5
            ):

                rx, ry, rw, rh = world.obstacle

                margin = OBSTACLE_MARGIN
                epsilon = 0.01

                top_y = (
                    ry + rh + margin + epsilon
                )

                bottom_y = (
                    ry - margin - epsilon
                )

                left_x = (
                    rx - margin - epsilon
                )

                right_x = (
                    rx + rw + margin + epsilon
                )

                above_1 = (
                    left_x,
                    top_y
                )

                above_2 = (
                    right_x,
                    top_y
                )

                below_1 = (
                    left_x,
                    bottom_y
                )

                below_2 = (
                    right_x,
                    bottom_y
                )

                above_valid = (
                    not segment_intersects_rect(
                        current_position,
                        above_1,
                        world.obstacle
                    )
                    and
                    not segment_intersects_rect(
                        above_1,
                        above_2,
                        world.obstacle
                    )
                    and
                    not segment_intersects_rect(
                        above_2,
                        predicted_target,
                        world.obstacle
                    )
                )

                below_valid = (
                    not segment_intersects_rect(
                        current_position,
                        below_1,
                        world.obstacle
                    )
                    and
                    not segment_intersects_rect(
                        below_1,
                        below_2,
                        world.obstacle
                    )
                    and
                    not segment_intersects_rect(
                        below_2,
                        predicted_target,
                        world.obstacle
                    )
                )

                print()
                print("NO_ROUTE DIAGNOSTIC")
                print("-" * 60)
                print(
                    f"Current position : "
                    f"({current_position[0]:.2f}, "
                    f"{current_position[1]:.2f})"
                )
                print(
                    f"Predicted target : "
                    f"({predicted_target[0]:.2f}, "
                    f"{predicted_target[1]:.2f})"
                )
                print(
                    f"Above valid      : "
                    f"{above_valid}"
                )
                print(
                    f"Below valid      : "
                    f"{below_valid}"
                )

                # Detailed corner diagnostics for remaining NO_ROUTE cases.
                if not above_valid and not below_valid:

                    print(
                        f"Obstacle         : "
                        f"({rx:.2f}, {ry:.2f}, "
                        f"{rw:.2f}, {rh:.2f})"
                    )

                    print(
                        f"Expanded corners:"
                    )

                    print(
                        f"  TL = ({left_x:.2f}, {top_y:.2f})"
                    )

                    print(
                        f"  TR = ({right_x:.2f}, {top_y:.2f})"
                    )

                    print(
                        f"  BL = ({left_x:.2f}, {bottom_y:.2f})"
                    )

                    print(
                        f"  BR = ({right_x:.2f}, {bottom_y:.2f})"
                    )

                    corners = {
                        "TL": above_1,
                        "TR": above_2,
                        "BL": below_1,
                        "BR": below_2
                    }

                    for corner_name, corner in corners.items():

                        start_clear = not segment_intersects_rect(
                            current_position,
                            corner,
                            world.obstacle
                        )

                        target_clear = not segment_intersects_rect(
                            corner,
                            predicted_target,
                            world.obstacle
                        )

                        print(
                            f"  {corner_name}: "
                            f"start={start_clear} "
                            f"target={target_clear}"
                        )

            waypoint_index = 0

        if (
            route_name != "NO_ROUTE"
            and
            waypoint_index < len(waypoints)
        ):

            destination = waypoints[
                waypoint_index
            ]

            waypoint_distance = np.sqrt(
                (
                    destination[0]
                    - current_position[0]
                ) ** 2
                +
                (
                    destination[1]
                    - current_position[1]
                ) ** 2
            )

            if waypoint_distance < 3.0:

                waypoint_index += 1

                if waypoint_index < len(waypoints):
                    destination = waypoints[
                        waypoint_index
                    ]
                else:
                    destination = actual_target

        else:
            destination = actual_target

        # Close-range correction is checked every controller step,
        # not only during neural updates. This prevents the fly from
        # chasing a stale predicted target when it is already close.
        live_target_distance = np.sqrt(
            (
                actual_target[0] - current_position[0]
            ) ** 2
            +
            (
                actual_target[1] - current_position[1]
            ) ** 2
        )

        if live_target_distance <= 15.0:
            destination = actual_target

        previous_position = current_position

        current_position = adaptive_step(
            current_position,
            destination,
            world.obstacle
        )

        # Detailed trajectory logging for the two failing baseline trials.
        if trial_index in (0, 9) and step % 5 == 0:
            print(
                f"  [FAIL-TRACE] trial={trial_index + 1:02d} "
                f"step={step:03d} "
                f"fly=({current_position[0]:.1f},"
                f"{current_position[1]:.1f}) "
                f"target=({world.target_x:.1f},"
                f"{world.target_y:.1f}) "
                f"dest=({destination[0]:.1f},"
                f"{destination[1]:.1f}) "
                f"route={route_name} "
                f"wp={waypoint_index} "
                f"dist={target_distance if 'target_distance' in locals() else 0:.2f}"
            )

        target_distance = np.sqrt(
            (
                world.target_x
                - current_position[0]
            ) ** 2
            +
            (
                world.target_y
                - current_position[1]
            ) ** 2
        )

        # Detailed diagnostics near difficult/failing trajectories.
        if (
            step % 10 == 0
            and
            target_distance > TARGET_THRESHOLD
        ):
            if (
                start_distance > 0
                and
                target_distance > start_distance * 0.75
            ):
                print(
                    f"  [TRACK] step={step:03d} "
                    f"fly=({current_position[0]:.1f},"
                    f"{current_position[1]:.1f}) "
                    f"target=({world.target_x:.1f},"
                    f"{world.target_y:.1f}) "
                    f"pred=({predicted_target[0]:.1f},"
                    f"{predicted_target[1]:.1f}) "
                    f"dist={target_distance:.2f} "
                    f"route={route_name}"
                )

        if (
            target_distance <= TARGET_THRESHOLD
        ):

            reduction = (
                1.0
                - target_distance / start_distance
            ) * 100.0

            return {
                "success": True,
                "steps": step + 1,
                "start_distance": start_distance,
                "final_distance": target_distance,
                "reduction": reduction,
                "route": route_name,
                "direct_blocked": direct_blocked,
                "route_history": route_history,
                "start_position": (
                    world.fly_x,
                    world.fly_y
                ),
                "start_target": start_target
            }

    final_distance = np.sqrt(
        (
            world.target_x
            - current_position[0]
        ) ** 2
        +
        (
            world.target_y
            - current_position[1]
        ) ** 2
    )

    reduction = (
        1.0
        - final_distance / start_distance
    ) * 100.0

    return {
        "success": False,
        "steps": MAX_STEPS,
        "start_distance": start_distance,
        "final_distance": final_distance,
        "reduction": reduction,
        "route": route_name,
        "direct_blocked": direct_blocked,
        "route_history": route_history,
        "start_position": (
            world.fly_x,
            world.fly_y
        ),
        "start_target": start_target
    }


def localization_test(
    brain,
    azimuth,
    decoder
):
    test_points = [
        (10.0, 10.0),
        (15.0, 50.0),
        (50.0, 50.0),
        (85.0, 50.0),
        (90.0, 90.0)
    ]

    print()
    print("=" * 75)
    print("NEURAL LOCALIZATION DIAGNOSTIC")
    print("=" * 75)

    print()
    print("Input -> Decoded -> Error")

    errors = []

    for x, y in test_points:

        decoded_x, decoded_y = neural_localize(
            brain,
            azimuth,
            decoder,
            x,
            y
        )

        error = np.sqrt(
            (
                decoded_x - x
            ) ** 2
            +
            (
                decoded_y - y
            ) ** 2
        )

        errors.append(error)

        print(
            f"({x:.0f},{y:.0f}) "
            f"-> "
            f"({decoded_x:.1f},{decoded_y:.1f}) "
            f"error {error:.2f}"
        )

    print()
    print(
        f"Mean error {np.mean(errors):.2f}"
    )

    print(
        f"Max error {np.max(errors):.2f}"
    )

    return errors


def benchmark(
    brain,
    azimuth,
    decoder
):
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    results = []

    print()
    print("=" * 75)
    print("BENCHMARK")
    print("=" * 75)

    for trial in range(TRIALS):

        result = run_trial(
            brain,
            azimuth,
            decoder,
            rng,
            trial_index=trial
        )

        results.append(result)

        status = (
            "SUCCESS"
            if result["success"]
            else "FAILED"
        )

        print(
            f"Trial {trial + 1:02d}/{TRIALS}: "
            f"{status} | "
            f"route={result['route']} | "
            f"direct={'BLOCKED' if result.get('direct_blocked', False) else 'CLEAR'} | "
            f"final={result['final_distance']:.2f} | "
            f"reduction={result['reduction']:.2f}%"
        )

    successes = sum(
        r["success"]
        for r in results
    )

    reductions = np.asarray(
        [
            r["reduction"]
            for r in results
        ],
        dtype=np.float32
    )

    final_distances = np.asarray(
        [
            r["final_distance"]
            for r in results
        ],
        dtype=np.float32
    )

    blocked_trials = sum(
        r.get("direct_blocked", False)
        for r in results
    )

    clear_trials = len(results) - blocked_trials

    route_history_counts = {
        "DIRECT": 0,
        "ABOVE": 0,
        "BELOW": 0,
        "NO_ROUTE": 0
    }

    blocked_route_counts = {
        "DIRECT": 0,
        "ABOVE": 0,
        "BELOW": 0,
        "NO_ROUTE": 0
    }

    clear_route_counts = {
        "DIRECT": 0,
        "ABOVE": 0,
        "BELOW": 0,
        "NO_ROUTE": 0
    }

    blocked_direct_decisions = 0

    for result in results:
        for (
            actual_blocked,
            neural_blocked,
            selected_route
        ) in result.get("route_history", []):

            route_history_counts[selected_route] = (
                route_history_counts.get(
                    selected_route,
                    0
                ) + 1
            )

            if actual_blocked:
                blocked_route_counts[selected_route] = (
                    blocked_route_counts.get(
                        selected_route,
                        0
                    ) + 1
                )
            else:
                clear_route_counts[selected_route] = (
                    clear_route_counts.get(
                        selected_route,
                        0
                    ) + 1
                )

            if (
                actual_blocked
                and
                selected_route == "DIRECT"
            ):
                blocked_direct_decisions += 1

    print()
    print("OBSTACLE DIAGNOSTIC")
    print("-" * 75)
    print(
        f"Trials with direct path BLOCKED : "
        f"{blocked_trials}"
    )
    print(
        f"Trials with direct path CLEAR   : "
        f"{clear_trials}"
    )

    print()
    print("ROUTE DECISIONS")
    print(
        f"  DIRECT   : "
        f"{route_history_counts['DIRECT']}"
    )
    print(
        f"  ABOVE    : "
        f"{route_history_counts['ABOVE']}"
    )
    print(
        f"  BELOW    : "
        f"{route_history_counts['BELOW']}"
    )
    print(
        f"  NO_ROUTE : "
        f"{route_history_counts['NO_ROUTE']}"
    )
    print(
        f"  BLOCKED + DIRECT : "
        f"{blocked_direct_decisions}"
    )

    print()
    print("BLOCKED-PATH ROUTES")
    print(
        f"  DIRECT   : {blocked_route_counts['DIRECT']}"
    )
    print(
        f"  ABOVE    : {blocked_route_counts['ABOVE']}"
    )
    print(
        f"  BELOW    : {blocked_route_counts['BELOW']}"
    )
    print(
        f"  NO_ROUTE : {blocked_route_counts['NO_ROUTE']}"
    )

    print()
    print("CLEAR-PATH ROUTES")
    print(
        f"  DIRECT   : {clear_route_counts['DIRECT']}"
    )
    print(
        f"  ABOVE    : {clear_route_counts['ABOVE']}"
    )
    print(
        f"  BELOW    : {clear_route_counts['BELOW']}"
    )
    print(
        f"  NO_ROUTE : {clear_route_counts['NO_ROUTE']}"
    )

    routes = {}

    for result in results:
        route = result["route"]

        routes[route] = (
            routes.get(route, 0)
            + 1
        )

    print()
    print("=" * 75)
    print("FINAL RESULTS")
    print("=" * 75)

    print()

    print(
        f"Successful {successes}/{TRIALS}"
    )

    print(
        f"Success rate "
        f"{successes / TRIALS * 100:.2f}%"
    )

    print(
        f"Mean reduction "
        f"{np.mean(reductions):.2f}%"
    )

    print(
        f"Median reduction "
        f"{np.median(reductions):.2f}%"
    )

    print(
        f"Mean final dist "
        f"{np.mean(final_distances):.2f}"
    )

    print(
        f"Median final dist "
        f"{np.median(final_distances):.2f}"
    )

    print(
        f"Best "
        f"{np.min(final_distances):.2f}"
    )

    print(
        f"Worst "
        f"{np.max(final_distances):.2f}"
    )

    print()
    print("Routes:")

    print(
        f"  DIRECT   "
        f"{routes.get('DIRECT', 0)}"
    )

    print(
        f"  ABOVE    "
        f"{routes.get('ABOVE', 0)}"
    )

    print(
        f"  BELOW    "
        f"{routes.get('BELOW', 0)}"
    )

    print(
        f"  NO_ROUTE "
        f"{routes.get('NO_ROUTE', 0)}"
    )

    return results


def main():

    print("=" * 75)
    print(
        "FLY-001 2D MOVING TARGET + OBSTACLE BENCHMARK"
    )
    print("=" * 75)

    print()
    print(f"Trials {TRIALS}")
    print(f"Max steps {MAX_STEPS}")
    print(f"Target threshold {TARGET_THRESHOLD}")
    print(f"Target speed {TARGET_SPEED}")
    print(
        f"Prediction horizon "
        f"{PREDICTION_HORIZON}"
    )
    print(
        f"Neural update every "
        f"{NEURAL_UPDATE_EVERY}"
    )
    print(
        f"Obstacle margin "
        f"{OBSTACLE_MARGIN}"
    )
    print(
        f"Random seed "
        f"{RANDOM_SEED}"
    )
    print(
        f"Neurons "
        f"{N_NEURONS}"
    )
    print(
        "Visual receptors 6006"
    )

    print()
    print(
        "Loading decoder..."
    )

    decoder = NeuralDecoder(
        MODEL_FILE
    )

    print(
        f"Decoder loaded "
        f"{MODEL_FILE}"
    )

    print()
    print(
        "Loading FlyBrain..."
    )

    brain = FlyBrain(
        device="auto"
    )

    azimuth = np.asarray(
        brain.azimuth,
        dtype=np.float32
    )

    print(
        f"Brain loaded "
        f"{N_NEURONS} neurons"
    )

    print(
        f"Visual receptors "
        f"{len(brain.visual)}"
    )

    route_planner_test()

    localization_test(
        brain,
        azimuth,
        decoder
    )

    benchmark(
        brain,
        azimuth,
        decoder
    )

    print()
    print("=" * 75)
    print("TEST COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()


