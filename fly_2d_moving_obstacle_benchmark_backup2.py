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


def plan_route(start, target, obstacle):
    rx, ry, rw, rh = obstacle

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

    if segment_clear(
        start,
        target,
        obstacle
    ):
        return "DIRECT", [target]

    above_1 = (
        left_x,
        top_y
    )

    above_2 = (
        right_x,
        top_y
    )

    above_valid = (
        segment_clear(
            start,
            above_1,
            obstacle
        )
        and
        segment_clear(
            above_1,
            above_2,
            obstacle
        )
        and
        segment_clear(
            above_2,
            target,
            obstacle
        )
    )

    if above_valid:
        return "ABOVE", [
            above_1,
            above_2,
            target
        ]

    below_1 = (
        left_x,
        bottom_y
    )

    below_2 = (
        right_x,
        bottom_y
    )

    below_valid = (
        segment_clear(
            start,
            below_1,
            obstacle
        )
        and
        segment_clear(
            below_1,
            below_2,
            obstacle
        )
        and
        segment_clear(
            below_2,
            target,
            obstacle
        )
    )

    if below_valid:
        return "BELOW", [
            below_1,
            below_2,
            target
        ]

    return "NO_ROUTE", []


def route_planner_test():
    start = (
        15.0,
        50.0
    )

    target = (
        85.0,
        50.0
    )

    obstacle = (
        45.0,
        35.0,
        10.0,
        30.0
    )

    route, waypoints = plan_route(
        start,
        target,
        obstacle
    )

    print()
    print("Route planner check:")
    print("  Expected : ABOVE or BELOW")
    print(f"  Actual   : {route}")

    if waypoints:
        print("  Waypoints:")

        for waypoint in waypoints:
            print(
                f"    ({waypoint[0]:.2f}, {waypoint[1]:.2f})"
            )

    return route


class MovingWorld:
    def __init__(self, rng):
        self.rng = rng

        self.obstacle = (
            45.0,
            35.0,
            10.0,
            30.0
        )

        while True:
            self.fly_x = float(
                rng.uniform(
                    5.0,
                    25.0
                )
            )

            self.fly_y = float(
                rng.uniform(
                    10.0,
                    90.0
                )
            )

            if not point_in_rect(
                (
                    self.fly_x,
                    self.fly_y
                ),
                self.obstacle
            ):
                break

        self.target_x = float(
            rng.uniform(
                75.0,
                95.0
            )
        )

        self.target_y = float(
            rng.uniform(
                10.0,
                90.0
            )
        )

        angle = float(
            rng.uniform(
                0.0,
                2.0 * np.pi
            )
        )

        self.target_vx = (
            TARGET_SPEED * np.cos(angle)
        )

        self.target_vy = (
            TARGET_SPEED * np.sin(angle)
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
    rng
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

            route_name, waypoints = plan_route(
                (
                    neural_x,
                    neural_y
                ),
                predicted_target,
                world.obstacle
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

        current_position = adaptive_step(
            current_position,
            destination,
            world.obstacle
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

        if target_distance <= TARGET_THRESHOLD:

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
                "route": route_name
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
        "route": route_name
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
            rng
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


