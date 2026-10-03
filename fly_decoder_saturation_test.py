import numpy as np
from flybrain import FlyBrain

MODEL_FILE = "fly_2d_continuous_decoder.npz"

SIGMA = 0.08
STEPS = 50
TRIALS = 3
N_NEURONS = 166700

POSITIONS = np.linspace(-1.0, 1.0, 21)


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

    # Project-level artificial X/Y channel split.
    stimulus[:midpoint] = signal_x[:midpoint]
    stimulus[midpoint:] = signal_y[midpoint:]

    return stimulus


def decode(activity, decoder):
    features_x = decoder["features_x"]
    features_y = decoder["features_y"]

    coef_x = decoder["coef_x"]
    coef_y = decoder["coef_y"]

    intercept_x = float(
        decoder["intercept_x"]
    )

    intercept_y = float(
        decoder["intercept_y"]
    )

    x_features = activity[features_x]
    y_features = activity[features_y]

    pred_x = float(
        x_features @ coef_x + intercept_x
    )

    pred_y = float(
        y_features @ coef_y + intercept_y
    )

    return pred_x, pred_y


def run_test(brain, azimuth, decoder, mode):

    results = []

    print()
    print("=" * 75)
    print(f"{mode} SATURATION TEST")
    print("=" * 75)

    print(
        f"{'TRUE':>8} | "
        f"{'DECODED':>10} | "
        f"{'ERROR':>10} | "
        f"{'TRIAL STD':>10}"
    )

    print("-" * 75)

    for position in POSITIONS:

        predictions = []

        for trial in range(TRIALS):

            if mode == "X":

                stimulus = encode_2d(
                    azimuth,
                    position,
                    0.0
                )

            elif mode == "Y":

                stimulus = encode_2d(
                    azimuth,
                    0.0,
                    position
                )

            elif mode == "DIAGONAL":

                stimulus = encode_2d(
                    azimuth,
                    position,
                    position
                )

            brain.reset()

            activity = np.zeros(
                N_NEURONS,
                dtype=np.float32
            )

            for _ in range(STEPS):

                fired = brain.step(
                    eye_drive=stimulus
                )

                if len(fired) > 0:
                    activity[fired] += 1

            pred_x, pred_y = decode(
                activity,
                decoder
            )

            if mode == "X":

                prediction = pred_x

            elif mode == "Y":

                prediction = pred_y

            else:

                prediction = (
                    pred_x + pred_y
                ) / 2.0

            predictions.append(
                prediction
            )

        predictions = np.asarray(
            predictions,
            dtype=np.float64
        )

        mean_prediction = predictions.mean()

        error = (
            mean_prediction - position
        )

        trial_std = predictions.std()

        print(
            f"{position:+8.2f} | "
            f"{mean_prediction:+10.3f} | "
            f"{error:+10.3f} | "
            f"{trial_std:10.3f}"
        )

        results.append(
            [
                position,
                mean_prediction,
                error,
                trial_std
            ]
        )

    return np.asarray(
        results,
        dtype=np.float64
    )


def summarize(results, name):

    true = results[:, 0]
    predicted = results[:, 1]

    errors = predicted - true

    mae = np.mean(
        np.abs(errors)
    )

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    correlation = np.corrcoef(
        true,
        predicted
    )[0, 1]

    worst_index = np.argmax(
        np.abs(errors)
    )

    print()
    print("-" * 60)
    print(f"{name} SUMMARY")
    print("-" * 60)

    print(
        f"MAE          : {mae:.4f}"
    )

    print(
        f"RMSE         : {rmse:.4f}"
    )

    print(
        f"Correlation  : {correlation:.4f}"
    )

    print(
        f"Max error    : "
        f"{errors[worst_index]:+.4f}"
    )

    print(
        f"Worst point  : "
        f"{true[worst_index]:+.2f}"
        f" -> "
        f"{predicted[worst_index]:+.3f}"
    )


def main():

    print("=" * 75)
    print("FLY-001 2D DECODER SATURATION TEST")
    print("=" * 75)

    print()
    print("Loading decoder...")

    decoder = np.load(
        MODEL_FILE,
        allow_pickle=False
    )

    print("Decoder keys:")
    print(decoder.files)

    print()
    print("Loading FlyBrain...")

    brain = FlyBrain(
        device="auto"
    )

    print(
        "Neurons  : "
        f"{N_NEURONS}"
    )

    print(
        "Visual   : "
        f"{len(brain.visual)}"
    )

    azimuth = np.asarray(
        brain.azimuth,
        dtype=np.float32
    )

    print(
        "Azimuth  : "
        f"{azimuth.min():.2f} "
        f"to "
        f"{azimuth.max():.2f}"
    )

    print(
        "Positions tested : "
        f"{len(POSITIONS)}"
    )

    print(
        "Trials per point : "
        f"{TRIALS}"
    )

    print(
        "Brain steps      : "
        f"{STEPS}"
    )

    # --------------------------------------------------
    # X AXIS
    # --------------------------------------------------

    x_results = run_test(
        brain,
        azimuth,
        decoder,
        "X"
    )

    summarize(
        x_results,
        "X AXIS"
    )

    # --------------------------------------------------
    # Y AXIS
    # --------------------------------------------------

    y_results = run_test(
        brain,
        azimuth,
        decoder,
        "Y"
    )

    summarize(
        y_results,
        "Y AXIS"
    )

    # --------------------------------------------------
    # DIAGONAL
    # --------------------------------------------------

    diagonal_results = run_test(
        brain,
        azimuth,
        decoder,
        "DIAGONAL"
    )

    summarize(
        diagonal_results,
        "DIAGONAL"
    )

    # --------------------------------------------------
    # FINAL
    # --------------------------------------------------

    print()
    print("=" * 75)
    print("TEST COMPLETE")
    print("=" * 75)

    print()
    print(
        "Important points to inspect:"
    )

    print(
        "  -1.00"
    )

    print(
        "  -0.90"
    )

    print(
        "  +0.90"
    )

    print(
        "  +1.00"
    )

    print()
    print(
        "If the decoder becomes strongly biased "
        "near +/-1.0, we will investigate edge "
        "saturation."
    )

    print()
    print(
        "If the decoder remains accurate across "
        "the whole range, the moving-target failure "
        "is likely coming from the navigation/controller "
        "rather than the decoder itself."
    )

    print()
    print(
        "No world movement was used."
    )

    print(
        "This test isolates the brain + decoder."
    )


if __name__ == "__main__":
    main()