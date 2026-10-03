import numpy as np
from flybrain import FlyBrain

MODEL_FILE = "fly_2d_continuous_decoder.npz"

SIGMA = 0.08
STEPS = 50
TRIALS = 6
N_NEURONS = 166700

TRAIN_POSITIONS = np.arange(-1.0, 1.01, 0.2)
TEST_POSITIONS = np.arange(-0.9, 0.91, 0.2)


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

    signal_x = soft_signal(
        azimuth,
        x_position
    )

    signal_y = soft_signal(
        azimuth,
        y_position
    )

    midpoint = len(azimuth) // 2

    stimulus = np.zeros(
        len(azimuth),
        dtype=np.float32
    )

    stimulus[:midpoint] = signal_x[:midpoint]
    stimulus[midpoint:] = signal_y[midpoint:]

    return stimulus


def collect_activity(brain, stimulus):

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

    return activity


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

    pred_x = float(
        activity[features_x] @ coef_x
        + intercept_x
    )

    pred_y = float(
        activity[features_y] @ coef_y
        + intercept_y
    )

    return pred_x, pred_y


def evaluate(brain, azimuth, decoder, positions, name):

    true_x = []
    true_y = []

    pred_x = []
    pred_y = []

    print()
    print("=" * 75)
    print(name)
    print("=" * 75)

    total = len(positions) ** 2 * TRIALS

    completed = 0

    for x_position in positions:

        for y_position in positions:

            stimulus = encode_2d(
                azimuth,
                x_position,
                y_position
            )

            for trial in range(TRIALS):

                activity = collect_activity(
                    brain,
                    stimulus
                )

                x, y = decode(
                    activity,
                    decoder
                )

                true_x.append(x_position)
                true_y.append(y_position)

                pred_x.append(x)
                pred_y.append(y)

                completed += 1

                if completed % 100 == 0:
                    print(
                        f"Progress: "
                        f"{completed}/{total}"
                    )

    true_x = np.asarray(true_x)
    true_y = np.asarray(true_y)

    pred_x = np.asarray(pred_x)
    pred_y = np.asarray(pred_y)

    x_error = pred_x - true_x
    y_error = pred_y - true_y

    x_mae = np.mean(
        np.abs(x_error)
    )

    y_mae = np.mean(
        np.abs(y_error)
    )

    x_rmse = np.sqrt(
        np.mean(x_error ** 2)
    )

    y_rmse = np.sqrt(
        np.mean(y_error ** 2)
    )

    x_corr = np.corrcoef(
        true_x,
        pred_x
    )[0, 1]

    y_corr = np.corrcoef(
        true_y,
        pred_y
    )[0, 1]

    print()
    print("-" * 60)
    print(f"{name} RESULTS")
    print("-" * 60)

    print()
    print("X DECODER")
    print(
        f"MAE         : {x_mae:.4f}"
    )
    print(
        f"RMSE        : {x_rmse:.4f}"
    )
    print(
        f"Correlation : {x_corr:.4f}"
    )

    print()
    print("Y DECODER")
    print(
        f"MAE         : {y_mae:.4f}"
    )
    print(
        f"RMSE        : {y_rmse:.4f}"
    )
    print(
        f"Correlation : {y_corr:.4f}"
    )

    return {
        "x_mae": x_mae,
        "y_mae": y_mae,
        "x_rmse": x_rmse,
        "y_rmse": y_rmse,
        "x_corr": x_corr,
        "y_corr": y_corr
    }


def main():

    print("=" * 75)
    print("FLY-001 2D DECODER REPRODUCTION TEST")
    print("=" * 75)

    print()
    print("Loading decoder...")

    decoder = np.load(
        MODEL_FILE,
        allow_pickle=False
    )

    print(
        "Decoder keys:",
        decoder.files
    )

    print()
    print("Loading FlyBrain...")

    brain = FlyBrain(
        device="auto"
    )

    azimuth = np.asarray(
        brain.azimuth,
        dtype=np.float32
    )

    print(
        f"Neurons : {N_NEURONS}"
    )

    print(
        f"Visual  : {len(brain.visual)}"
    )

    print(
        f"Azimuth : "
        f"{azimuth.min():.2f} "
        f"to "
        f"{azimuth.max():.2f}"
    )

    print()
    print("Training grid:")
    print(
        TRAIN_POSITIONS
    )

    print()
    print("Held-out grid:")
    print(
        TEST_POSITIONS
    )

    print()
    print(
        f"Trials per grid point : {TRIALS}"
    )

    print(
        f"Brain steps per sample: {STEPS}"
    )

    # --------------------------------------------------
    # TRAIN GRID REPRODUCTION
    # --------------------------------------------------

    train_results = evaluate(
        brain,
        azimuth,
        decoder,
        TRAIN_POSITIONS,
        "TRAINING GRID REPRODUCTION"
    )

    # --------------------------------------------------
    # HELD-OUT GRID REPRODUCTION
    # --------------------------------------------------

    test_results = evaluate(
        brain,
        azimuth,
        decoder,
        TEST_POSITIONS,
        "HELD-OUT GRID REPRODUCTION"
    )

    # --------------------------------------------------
    # FINAL COMPARISON
    # --------------------------------------------------

    print()
    print("=" * 75)
    print("FINAL COMPARISON")
    print("=" * 75)

    print()
    print("Previously reported:")
    print(
        "X held-out correlation : 0.9848"
    )
    print(
        "Y held-out correlation : 0.9800"
    )

    print()
    print("This reproduction:")
    print(
        f"X held-out correlation : "
        f"{test_results['x_corr']:.4f}"
    )

    print(
        f"Y held-out correlation : "
        f"{test_results['y_corr']:.4f}"
    )

    print()
    print(
        "If these are close to 0.98, "
        "the decoder is behaving as expected "
        "and the saturation test used a different "
        "input condition."
    )

    print()
    print(
        "If these are also poor, "
        "we need to investigate the decoder "
        "training/validation pipeline itself."
    )

    print()
    print("TEST COMPLETE")


if __name__ == "__main__":
    main()