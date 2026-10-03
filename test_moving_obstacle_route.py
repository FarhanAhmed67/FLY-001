OBSTACLE_MARGIN = 4.0


def segment_hits_rect(p1, p2, rect, samples=200):

    x1, y1 = p1
    x2, y2 = p2

    rx, ry, rw, rh = rect

    for i in range(samples + 1):

        t = i / samples

        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t

        if (
            rx <= x <= rx + rw
            and
            ry <= y <= ry + rh
        ):
            return True

    return False


def inflate_obstacle(obstacle, margin):

    x, y, w, h = obstacle

    return (
        x - margin,
        y - margin,
        w + 2 * margin,
        h + 2 * margin
    )


def plan_route(start, target, obstacle):

    inflated = inflate_obstacle(
        obstacle,
        OBSTACLE_MARGIN
    )

    if not segment_hits_rect(
        start,
        target,
        inflated
    ):

        return "DIRECT"

    rx, ry, rw, rh = inflated

    top_y = ry + rh + 0.5
    bottom_y = ry - 0.5

    top_left = (rx - 0.5, top_y)
    top_right = (rx + rw + 0.5, top_y)

    bottom_left = (rx - 0.5, bottom_y)
    bottom_right = (rx + rw + 0.5, bottom_y)

    above = (
        not segment_hits_rect(start, top_left, inflated)
        and
        not segment_hits_rect(top_left, top_right, inflated)
        and
        not segment_hits_rect(top_right, target, inflated)
    )

    if above:
        return "ABOVE"

    below = (
        not segment_hits_rect(start, bottom_left, inflated)
        and
        not segment_hits_rect(bottom_left, bottom_right, inflated)
        and
        not segment_hits_rect(bottom_right, target, inflated)
    )

    if below:
        return "BELOW"

    return "NO_ROUTE"


obstacle = (45.0, 35.0, 10.0, 30.0)

start = (15.0, 50.0)

target = (85.0, 50.0)

print("Obstacle :", obstacle)
print("Start    :", start)
print("Target   :", target)

print()
print("Route    :", plan_route(
    start,
    target,
    obstacle
))