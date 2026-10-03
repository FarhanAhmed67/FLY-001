import numpy as np
from flybrain import FlyBrain


SIGMA = 0.08
STEPS = 50
TRIALS = 3


def soft_signal(azimuth, position, sigma=SIGMA):
    azimuth = np.asarray(azimuth, dtype=np.float32)

    distance = np.abs(azimuth - position)

    signal = np.exp(
        -(distance ** 2) / (2.0 * sigma ** 2)
    )

    signal[~np.isfinite(azimuth)] = 0.0

    maximum = signal.max()

    if maximum > 0:
        signal /= maximum

    return signal.astype(np.float32)


def make_2d_input(azimuth, x_position, y_position):
    n = len(azimuth)
    half = n // 2

    horizontal = soft_signal(
        azimuth,
        x_position
    )

    vertical = soft_signal(
        azimuth,
        y_position
    )

    stimulus = np.zeros(n, dtype=np.float32)

    stimulus[:half] = horizontal[:half]
    stimulus[half:] = vertical[half:]

    return stimulus


def predict(activity, features, coef, intercept):
    selected = activity[features]
    return float(selected @ coef + intercept)


print("=" * 70)
print("FLY-001 2D DECODER VALIDATION")
print("=" * 70)

print()
print("Loading FlyBrain...")

brain = FlyBrain(device="auto")

print(f"Neurons   : {len(brain.cell_type)}")
print(f"Receptors : {len(brain.visual)}")
print(f"Azimuth   : {brain.azimuth.min():.2f} to {brain.azimuth.max():.2f}")

print()
print("Loading corrected decoder...")

model = np.load(
    "fly_2d_continuous_decoder.npz"
)

features_x = model["features_x"]
features_y = model["features_y"]

coef_x = model["coef_x"]
coef_y = model["coef_y"]

intercept_x = float(model["intercept_x"])
intercept_y = float(model["intercept_y"])

print(f"X features : {len(features_x)}")
print(f"Y features : {len(features_y)}")

test_values = np.arange(
    -0.9,
    0.91,
    0.2
)

print()
print("Testing unseen positions...")
print(test_values)

errors_x = []
errors_y = []

predictions_x = []
predictions_y = []

true_x_values = []
true_y_values = []

for x_pos in test_values:

    for y_pos in test_values:

        for trial in range(TRIALS):

            brain.reset()

            stimulus = make_2d_input(
                brain.azimuth,
                x_pos,
                y_pos
            )

            activity = np.zeros(
                len(brain.cell_type),
                dtype=np.float32
            )

            for _ in range(STEPS):

                fired = brain.step(
                    eye_drive=stimulus
                )

                if len(fired) > 0:
                    activity[fired] += 1

            pred_x = predict(
                activity,
                features_x,
                coef_x,
                intercept_x
            )

            pred_y = predict(
                activity,
                features_y,
                coef_y,
                intercept_y
            )

            predictions_x.append(pred_x)
            predictions_y.append(pred_y)

            true_x_values.append(x_pos)
            true_y_values.append(y_pos)

            errors_x.append(abs(pred_x - x_pos))
            errors_y.append(abs(pred_y - y_pos))


predictions_x = np.asarray(predictions_x)
predictions_y = np.asarray(predictions_y)

true_x_values = np.asarray(true_x_values)
true_y_values = np.asarray(true_y_values)

errors_x = np.asarray(errors_x)
errors_y = np.asarray(errors_y)


corr_x = np.corrcoef(
    predictions_x,
    true_x_values
)[0, 1]

corr_y = np.corrcoef(
    predictions_y,
    true_y_values
)[0, 1]


rmse_x = np.sqrt(
    np.mean(
        (predictions_x - true_x_values) ** 2
    )
)

rmse_y = np.sqrt(
    np.mean(
        (predictions_y - true_y_values) ** 2
    )
)


print()
print("=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

print()
print("X DECODER")
print("-" * 50)
print(f"MAE  : {errors_x.mean():.4f}")
print(f"RMSE : {rmse_x:.4f}")
print(f"Corr : {corr_x:.4f}")

print()
print("Y DECODER")
print("-" * 50)
print(f"MAE  : {errors_y.mean():.4f}")
print(f"RMSE : {rmse_y:.4f}")
print(f"Corr : {corr_y:.4f}")


print()
print("=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

for i in range(min(20, len(predictions_x))):

    print(
        f"TRUE "
        f"({true_x_values[i]:+.2f}, {true_y_values[i]:+.2f})"
        f"  ->  "
        f"DECODED "
        f"({predictions_x[i]:+.2f}, {predictions_y[i]:+.2f})"
    )


print()
print("=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)