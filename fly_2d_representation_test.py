import numpy as np
from flybrain import FlyBrain


SIGMA = 0.08
BRAIN_STEPS = 50
TRIALS = 8

POSITIONS = np.arange(
    -1.0,
    1.01,
    0.2
)


def soft_signal(azimuth, position):

    distance = np.abs(
        azimuth - position
    )

    signal = np.exp(
        -(distance ** 2)
        / (2.0 * SIGMA ** 2)
    )

    maximum = signal.max()

    if maximum > 0:
        signal /= maximum

    return signal.astype(
        np.float32
    )


def collect_sample(
    brain,
    stimulus
):

    brain.reset()

    counts = np.zeros(
        len(brain.cell_type),
        dtype=np.float32
    )

    for _ in range(BRAIN_STEPS):

        fired = brain.step(
            eye_drive=stimulus
        )

        if len(fired) > 0:

            counts += np.bincount(
                fired,
                minlength=len(brain.cell_type)
            )

    return counts


def collect_dataset(
    brain,
    mode
):

    X = []
    y = []

    for position in POSITIONS:

        print(
            f"{mode.upper()} position "
            f"{position:+.2f}"
        )

        signal = soft_signal(
            brain.azimuth,
            position
        )

        for trial in range(TRIALS):

            if mode == "HORIZONTAL":

                stimulus = signal.copy()

            else:

                stimulus = np.zeros_like(
                    signal
                )

                # Use a different fixed
                # receptor subset for the
                # second channel.

                half = len(signal) // 2

                stimulus[half:] = (
                    signal[half:]
                )

            sample = collect_sample(
                brain,
                stimulus
            )

            X.append(sample)
            y.append(position)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32)
    )


def select_features(
    X,
    y,
    count=1000
):

    overall_mean = X.mean(
        axis=0
    )

    scores = np.zeros(
        X.shape[1],
        dtype=np.float32
    )

    for position in np.unique(y):

        mask = np.isclose(
            y,
            position
        )

        class_mean = X[
            mask
        ].mean(axis=0)

        scores += (
            np.sum(mask)
            * (
                class_mean
                - overall_mean
            ) ** 2
        )

    count = min(
        count,
        X.shape[1]
    )

    return np.argpartition(
        scores,
        -count
    )[-count:]


def train_decoder(
    X,
    y
):

    features = select_features(
        X,
        y
    )

    X = X[:, features]

    mean = X.mean(
        axis=0
    )

    std = X.std(
        axis=0
    )

    std[std < 1e-6] = 1.0

    X = (
        X - mean
    ) / std

    alpha = 10.0

    A = (
        X.T @ X
        + alpha * np.eye(
            X.shape[1]
        )
    )

    b = X.T @ y

    weights = np.linalg.solve(
        A,
        b
    )

    prediction = X @ weights

    mae = np.mean(
        np.abs(
            prediction - y
        )
    )

    correlation = np.corrcoef(
        prediction,
        y
    )[0, 1]

    return (
        mae,
        correlation
    )


print()
print("=" * 65)
print("FLY-001 2D REPRESENTATION TEST")
print("=" * 65)

brain = FlyBrain(
    device="auto"
)

print()
print(
    "Neurons          :",
    len(brain.cell_type)
)

print(
    "Visual receptors :",
    len(brain.azimuth)
)


print()
print("=" * 65)
print("HORIZONTAL CHANNEL")
print("=" * 65)

X_horizontal, y_horizontal = (
    collect_dataset(
        brain,
        "HORIZONTAL"
    )
)

mae_h, corr_h = train_decoder(
    X_horizontal,
    y_horizontal
)

print()
print(
    f"Horizontal MAE         : "
    f"{mae_h:.4f}"
)

print(
    f"Horizontal correlation : "
    f"{corr_h:.4f}"
)


print()
print("=" * 65)
print("VERTICAL CHANNEL")
print("=" * 65)

X_vertical, y_vertical = (
    collect_dataset(
        brain,
        "VERTICAL"
    )
)

mae_v, corr_v = train_decoder(
    X_vertical,
    y_vertical
)

print()
print(
    f"Vertical MAE         : "
    f"{mae_v:.4f}"
)

print(
    f"Vertical correlation : "
    f"{corr_v:.4f}"
)


print()
print("=" * 65)
print("2D REPRESENTATION TEST COMPLETE")
print("=" * 65)