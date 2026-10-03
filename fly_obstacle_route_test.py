import math
import numpy as np


WORLD_WIDTH = 100.0
WORLD_HEIGHT = 100.0
MARGIN = 4.0

SEED = 42
TESTS = 20


def point_inside(x, y, obstacle, margin=0.0):

    return (
        obstacle["x"] - margin <= x <= obstacle["x"] + obstacle["width"] + margin
        and
        obstacle["y"] - margin <= y <= obstacle["y"] + obstacle["height"] + margin
    )


def line_hits(x1, y1, x2, y2, obstacle, margin=0.0):

    steps = 200

    for i in range(steps + 1):

        t = i / steps

        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t

        if point_inside(x, y, obstacle, margin):
            return True

    return False


def choose_route(fly_x, fly_y, target_x, target_y, obstacle):

    # --------------------------------------------------------
    # Check direct route.
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
        return "DIRECT", None

    # --------------------------------------------------------
    # Create inflated obstacle.
    # --------------------------------------------------------

    left = obstacle["x"] - MARGIN
    right = obstacle["x"] + obstacle["width"] + MARGIN
    bottom = obstacle["y"] - MARGIN
    top = obstacle["y"] + obstacle["height"] + MARGIN

    # --------------------------------------------------------
    # Four corners.
    # --------------------------------------------------------

    top_left = (left, top)
    top_right = (right, top)

    bottom_left = (left, bottom)
    bottom_right = (right, bottom)

    # --------------------------------------------------------
    # Since fly starts on the left and target is generally
    # on the right, test two complete routes.
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
    # Verify routes.
    # --------------------------------------------------------

    above_valid = (
        not line_hits(
            fly_x,
            fly_y,
            top_left[0],
            top_left[1],
            obstacle,
            0.0
        )
        and
        not line_hits(
            top_left[0],
            top_left[1],
            top_right[0],
            top_right[1],
            obstacle,
            0.0
        )
        and
        not line_hits(
            top_right[0],
            top_right[1],
            target_x,
            target_y,
            obstacle,
            0.0
        )
    )

    below_valid = (
        not line_hits(
            fly_x,
            fly_y,
            bottom_left[0],
            bottom_left[1],
            obstacle,
            0.0
        )
        and
        not line_hits(
            bottom_left[0],
            bottom_left[1],
            bottom_right[0],
            bottom_right[1],
            obstacle,
            0.0
        )
        and
        not line_hits(
            bottom_right[0],
            bottom_right[1],
            target_x,
            target_y,
            obstacle,
            0.0
        )
    )

    if above_valid and below_valid:

        if above_length <= below_length:

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

    return "NO_ROUTE", None


# ============================================================
# TEST WORLDS
# ============================================================

rng = np.random.default_rng(SEED)

direct_count = 0
above_count = 0
below_count = 0
no_route_count = 0

print()
print("=" * 70)
print("FLY-001 OBSTACLE ROUTE SANITY TEST")
print("=" * 70)
print()

for i in range(1, TESTS + 1):

    while True:

        fly_x = rng.uniform(5, 25)
        fly_y = rng.uniform(10, 90)

        target_x = rng.uniform(75, 95)
        target_y = rng.uniform(10, 90)

        obstacle = {
            "x": rng.uniform(35, 55),
            "y": rng.uniform(20, 55),
            "width": rng.uniform(8, 14),
            "height": rng.uniform(20, 32)
        }

        if point_inside(
            fly_x,
            fly_y,
            obstacle,
            MARGIN
        ):
            continue

        if point_inside(
            target_x,
            target_y,
            obstacle,
            MARGIN
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

        break

    route, waypoints = choose_route(
        fly_x,
        fly_y,
        target_x,
        target_y,
        obstacle
    )

    if route == "DIRECT":
        direct_count += 1

    elif route == "ABOVE":
        above_count += 1

    elif route == "BELOW":
        below_count += 1

    else:
        no_route_count += 1

    print(
        f"Test {i:02d} | "
        f"Route: {route:8s} | "
        f"Fly ({fly_x:5.1f},{fly_y:5.1f}) | "
        f"Target ({target_x:5.1f},{target_y:5.1f})"
    )


print()
print("=" * 70)
print("ROUTE TEST SUMMARY")
print("=" * 70)

print(f"DIRECT   : {direct_count}")
print(f"ABOVE    : {above_count}")
print(f"BELOW    : {below_count}")
print(f"NO ROUTE : {no_route_count}")

print("=" * 70)