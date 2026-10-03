import numpy as np
from flybrain import FlyBrain
from sklearn.linear_model import Ridge


# ============================================================
# CONFIGURATION
# ============================================================

SIGMA = 0.08
STEPS = 50
TRIALS = 6

TRAIN_VALUES = np.arange(-1.0, 1.01, 0.2)
TEST_VALUES = np.arange(-0.9, 0.91, 0.2)

FEATURES = 1000
RIDGE_ALPHA = 10.0

N_NEURONS = 166700


# ============================================================
# SOFT VISUAL ENCODER
# ============================================================

def soft_signal(azimuth, position, sigma=SIGMA):

    azimuth = np.asarray(
        azimuth,
        dtype=np.float32
    )

    distance = np.abs(
        azimuth - position
    )

    signal = np.exp(
        -(distance ** 2)
        / (2.0 * sigma ** 2)
    )

    signal[~np.isfinite(azimuth)] = 0.0

    maximum = signal.max()

    if maximum > 0:
        signal /= maximum

    return signal.astype(
        np.float32
    )


def make_2d_input(
    azimuth,
    x_position,
    y_position
):

    """
    Project-level artificial 2D representation.

    First half of receptors = X channel.
    Second half of receptors = Y channel.

    IMPORTANT:
    This uses the actual FlyBrain azimuth array.
    """

    x_signal = soft_signal(
        azimuth,
        x_position
    )

    y_signal = soft_signal(
        azimuth,
        y_position
    )

    n = len(azimuth)

    midpoint = n // 2

    stimulus = np.zeros(
        n,
        dtype=np.float32
    )

    stimulus[:midpoint] = (
        x_signal[:midpoint]
    )

    stimulus[midpoint:] = (
        y_signal[midpoint:]
    )

    return stimulus


# ============================================================
# COLLECT NEURAL ACTIVITY
# ============================================================

def collect_samples(
    brain,
    azimuth,
    positions,
    trials
):

    X = []
    y_x = []
    y_y = []

    total = (
        len(positions)
        * len(positions)
        * trials
    )

    completed = 0

    for x_pos in positions:

        for y_pos in positions:

            print(
                f"Collecting "
                f"({x_pos:+.1f}, {y_pos:+.1f})"
            )

            stimulus = make_2d_input(
                azimuth,
                x_pos,
                y_pos
            )

            for trial in range(trials):

                brain.reset()

                counts = np.zeros(
                    N_NEURONS,
                    dtype=np.float32
                )

                for _ in range(STEPS):

                    fired = brain.step(
                        eye_drive=stimulus
                    )

                    if len(fired) > 0:

                        counts[fired] += 1

                X.append(counts)

                y_x.append(x_pos)
                y_y.append(y_pos)

                completed += 1

                if completed % 100 == 0:

                    print(
                        f"Progress: "
                        f"{completed}/{total}"
                    )

    return (
        np.asarray(
            X,
            dtype=np.float32
        ),
        np.asarray(y_x),
        np.asarray(y_y)
    )


# ============================================================
# FEATURE SELECTION
# ============================================================

def select_features(
    X,
    y,
    count=FEATURES
):

    overall_mean = X.mean(
        axis=0
    )

    scores = np.zeros(
        X.shape[1],
        dtype=np.float32
    )

    for value in np.unique(y):

        mask = np.isclose(
            y,
            value
        )

        if not np.any(mask):
            continue

        class_mean = X[mask].mean(
            axis=0
        )

        scores += np.sum(mask) * (
            class_mean - overall_mean
        ) ** 2

    count = min(
        count,
        X.shape[1]
    )

    features = np.argpartition(
        scores,
        -count
    )[-count:]

    return features


# ============================================================
# TRAIN RIDGE DECODER
# ============================================================

def train_decoder(
    X,
    y,
    features
):

    X_selected = X[
        :,
        features
    ]

    model = Ridge(
        alpha=RIDGE_ALPHA
    )

    model.fit(
        X_selected,
        y
    )

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    X,
    y,
    features
):

    predictions = model.predict(
        X[:, features]
    )

    errors = predictions - y

    mae = np.mean(
        np.abs(errors)
    )

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    if (
        np.std(predictions) > 0
        and np.std(y) > 0
    ):

        corr = np.corrcoef(
            predictions,
            y
        )[0, 1]

    else:

        corr = 0.0

    return (
        predictions,
        mae,
        rmse,
        corr
    )


# ============================================================
# START
# ============================================================

print()
print("=" * 70)
print("FLY-001 CORRECTED 2D CONTINUOUS DECODER")
print("=" * 70)

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
    f"Neurons   : {len(brain.positions)}"
)

print(
    f"Receptors : {len(brain.visual)}"
)

print(
    f"Azimuth   : "
    f"{azimuth.min():.2f} "
    f"to "
    f"{azimuth.max():.2f}"
)

print(
    f"Unique azimuth values : "
    f"{len(np.unique(azimuth))}"
)

print()
print("Training grid:")
print(TRAIN_VALUES)

print()
print("Held-out grid:")
print(TEST_VALUES)

print()
print(
    f"Trials per point : {TRIALS}"
)

print(
    f"Brain steps      : {STEPS}"
)

print(
    f"Features         : {FEATURES}"
)

print(
    f"Ridge alpha      : {RIDGE_ALPHA}"
)


# ============================================================
# TRAINING DATA
# ============================================================

print()
print("=" * 70)
print("COLLECTING TRAINING DATA")
print("=" * 70)

X_train, yx_train, yy_train = collect_samples(
    brain,
    azimuth,
    TRAIN_VALUES,
    TRIALS
)

print()
print(
    f"Training samples: "
    f"{len(X_train)}"
)


# ============================================================
# FEATURE SELECTION
# ============================================================

print()
print("Selecting X features...")

features_x = select_features(
    X_train,
    yx_train
)

print(
    f"X features selected: "
    f"{len(features_x)}"
)

print()
print("Selecting Y features...")

features_y = select_features(
    X_train,
    yy_train
)

print(
    f"Y features selected: "
    f"{len(features_y)}"
)


# ============================================================
# TRAIN DECODERS
# ============================================================

print()
print("Training X decoder...")

decoder_x = train_decoder(
    X_train,
    yx_train,
    features_x
)

print()
print("Training Y decoder...")

decoder_y = train_decoder(
    X_train,
    yy_train,
    features_y
)


# ============================================================
# HELD-OUT DATA
# ============================================================

print()
print("=" * 70)
print("COLLECTING HELD-OUT DATA")
print("=" * 70)

X_test, yx_test, yy_test = collect_samples(
    brain,
    azimuth,
    TEST_VALUES,
    TRIALS
)


# ============================================================
# EVALUATE X
# ============================================================

print()
print("Evaluating X decoder...")

(
    pred_x_train,
    mae_x_train,
    rmse_x_train,
    corr_x_train
) = evaluate(
    decoder_x,
    X_train,
    yx_train,
    features_x
)

(
    pred_x_test,
    mae_x_test,
    rmse_x_test,
    corr_x_test
) = evaluate(
    decoder_x,
    X_test,
    yx_test,
    features_x
)


# ============================================================
# EVALUATE Y
# ============================================================

print()
print("Evaluating Y decoder...")

(
    pred_y_train,
    mae_y_train,
    rmse_y_train,
    corr_y_train
) = evaluate(
    decoder_y,
    X_train,
    yy_train,
    features_y
)

(
    pred_y_test,
    mae_y_test,
    rmse_y_test,
    corr_y_test
) = evaluate(
    decoder_y,
    X_test,
    yy_test,
    features_y
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("CORRECTED RESULTS")
print("=" * 70)

print()
print("X DECODER")
print("-" * 45)

print(
    f"Training MAE : "
    f"{mae_x_train:.4f}"
)

print(
    f"Training RMSE: "
    f"{rmse_x_train:.4f}"
)

print(
    f"Training Corr: "
    f"{corr_x_train:.4f}"
)

print()

print(
    f"Held-out MAE : "
    f"{mae_x_test:.4f}"
)

print(
    f"Held-out RMSE: "
    f"{rmse_x_test:.4f}"
)

print(
    f"Held-out Corr: "
    f"{corr_x_test:.4f}"
)


print()
print("Y DECODER")
print("-" * 45)

print(
    f"Training MAE : "
    f"{mae_y_train:.4f}"
)

print(
    f"Training RMSE: "
    f"{rmse_y_train:.4f}"
)

print(
    f"Training Corr: "
    f"{corr_y_train:.4f}"
)

print()

print(
    f"Held-out MAE : "
    f"{mae_y_test:.4f}"
)

print(
    f"Held-out RMSE: "
    f"{rmse_y_test:.4f}"
)

print(
    f"Held-out Corr: "
    f"{corr_y_test:.4f}"
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("SAMPLE HELD-OUT PREDICTIONS")
print("=" * 70)

for i in range(
    min(20, len(X_test))
):

    print(
        f"TRUE "
        f"({yx_test[i]:+.2f}, "
        f"{yy_test[i]:+.2f})"
        f"  ->  "
        f"DECODED "
        f"({pred_x_test[i]:+.2f}, "
        f"{pred_y_test[i]:+.2f})"
    )


# ============================================================
# SAVE CORRECTED MODEL
# ============================================================

np.savez(
    "fly_2d_continuous_decoder.npz",

    features_x=features_x,
    features_y=features_y,

    coef_x=decoder_x.coef_,
    intercept_x=decoder_x.intercept_,

    coef_y=decoder_y.coef_,
    intercept_y=decoder_y.intercept_,

    sigma=np.array(
        SIGMA,
        dtype=np.float32
    ),

    steps=np.array(
        STEPS,
        dtype=np.int32
    ),

    n_receptors=np.array(
        len(brain.visual),
        dtype=np.int32
    ),

    encoder_type=np.array(
        "actual_brain_azimuth_soft_2d"
    )
)


print()
print("=" * 70)
print("MODEL SAVED")
print("=" * 70)

print()
print(
    "fly_2d_continuous_decoder.npz"
)

print()
print(
    "Encoder: actual FlyBrain azimuth"
)

print(
    "X channel: first half"
)

print(
    "Y channel: second half"
)

print()
print("=" * 70)
print("CORRECTED 2D DECODER COMPLETE")
print("=" * 70)