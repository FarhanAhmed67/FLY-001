import numpy as np
from flybrain import FlyBrain


# =========================
# SETTINGS
# =========================

POSITIONS = {
    "LEFT": -0.75,
    "CENTER_LEFT": -0.375,
    "CENTER": 0.0,
    "CENTER_RIGHT": 0.375,
    "RIGHT": 0.75,
}

TRIALS_PER_POSITION = 20
MEASUREMENT_STEPS = 50       # 50 x 20 ms = 1 second
RECEPTORS_PER_STIMULUS = 500
INTENSITY = 1.0

TOP_FEATURES = 1000
FOLDS = 5

rng = np.random.default_rng(42)


# =========================
# BRAIN
# =========================

brain = FlyBrain(device="auto")

azimuth = np.asarray(brain.azimuth)
n_neurons = len(brain.cell_type)


# =========================
# CREATE STIMULUS
# =========================

def make_stimulus(direction):

    distance = np.abs(azimuth - direction)

    valid = np.where(np.isfinite(distance))[0]

    selected = valid[
        np.argsort(distance[valid])[:RECEPTORS_PER_STIMULUS]
    ]

    stimulus = np.zeros(len(azimuth), dtype=np.float32)
    stimulus[selected] = INTENSITY

    return stimulus


# =========================
# RUN ONE MEASUREMENT
# =========================

def measure():

    counts = np.zeros(n_neurons, dtype=np.float32)

    for _ in range(MEASUREMENT_STEPS):

        fired = brain.step(
            eye_drive=current_stimulus
        )

        fired = np.asarray(fired, dtype=np.int64)

        if len(fired) > 0:
            counts += np.bincount(
                fired,
                minlength=n_neurons
            )[:n_neurons]

    return counts / MEASUREMENT_STEPS


# =========================
# COLLECT DATA
# =========================

X_stim = []
X_off1 = []
X_off5 = []
labels = []


print()
print("=" * 70)
print("POSITION DECODER DATA COLLECTION")
print("=" * 70)
print()

for label, direction in POSITIONS.items():

    print(f"Collecting {label} ({direction:+.3f})...")

    stimulus = make_stimulus(direction)

    for trial in range(TRIALS_PER_POSITION):

        brain.reset()

        # -------------------------
        # STIMULATED
        # -------------------------

        current_stimulus = stimulus

        stimulated = measure()

        # -------------------------
        # OFF 1 SECOND
        # -------------------------

        current_stimulus = np.zeros_like(stimulus)

        off1 = measure()

        # -------------------------
        # OFF 5 SECONDS
        # -------------------------

        for _ in range(4):

            measure()

        off5 = measure()

        X_stim.append(stimulated)
        X_off1.append(off1)
        X_off5.append(off5)

        labels.append(label)

        print(
            f"  Trial {trial + 1:02d}/{TRIALS_PER_POSITION}",
            end="\r"
        )

    print()


X_stim = np.asarray(X_stim)
X_off1 = np.asarray(X_off1)
X_off5 = np.asarray(X_off5)
labels = np.asarray(labels)

print()
print("Data collection complete.")
print("Trials:", len(labels))
print("Neurons:", n_neurons)


# ============================================================
# CLASSIFIER
# ============================================================

classes = list(POSITIONS.keys())


def cosine_similarity(a, b):

    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)

    a_norm[a_norm == 0] = 1
    b_norm[b_norm == 0] = 1

    a = a / a_norm
    b = b / b_norm

    return a @ b.T


def select_features(X, y, top_k):

    overall_mean = X.mean(axis=0)

    scores = np.zeros(X.shape[1])

    for cls in classes:

        mask = y == cls

        cls_data = X[mask]

        if len(cls_data) == 0:
            continue

        cls_mean = cls_data.mean(axis=0)

        scores += len(cls_data) * (
            cls_mean - overall_mean
        ) ** 2

    top_k = min(top_k, X.shape[1])

    indices = np.argpartition(
        scores,
        -top_k
    )[-top_k:]

    return indices


def classify(train_X, train_y, test_X):

    centroids = []

    for cls in classes:

        centroid = train_X[
            train_y == cls
        ].mean(axis=0)

        centroids.append(centroid)

    centroids = np.asarray(centroids)

    similarities = cosine_similarity(
        test_X,
        centroids
    )

    predictions = np.asarray(classes)[
        np.argmax(similarities, axis=1)
    ]

    return predictions


def cross_validate(X, name):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    indices = np.arange(len(labels))

    accuracies = []

    shuffled = indices.copy()
    rng.shuffle(shuffled)

    fold_indices = np.array_split(
        shuffled,
        FOLDS
    )

    for fold in range(FOLDS):

        test_idx = fold_indices[fold]

        train_idx = np.concatenate(
            [
                fold_indices[i]
                for i in range(FOLDS)
                if i != fold
            ]
        )

        train_X = X[train_idx]
        train_y = labels[train_idx]

        test_X = X[test_idx]
        test_y = labels[test_idx]

        # Feature selection ONLY on training data.
        features = select_features(
            train_X,
            train_y,
            TOP_FEATURES
        )

        train_selected = train_X[:, features]
        test_selected = test_X[:, features]

        # Standardize using training statistics only.
        mean = train_selected.mean(axis=0)
        std = train_selected.std(axis=0)

        std[std < 1e-6] = 1.0

        train_selected = (
            train_selected - mean
        ) / std

        test_selected = (
            test_selected - mean
        ) / std

        predictions = classify(
            train_selected,
            train_y,
            test_selected
        )

        accuracy = np.mean(
            predictions == test_y
        )

        accuracies.append(accuracy)

        print(
            f"Fold {fold + 1}: "
            f"{accuracy * 100:.1f}%"
        )

    accuracies = np.asarray(accuracies)

    print()
    print(
        f"Average accuracy: "
        f"{accuracies.mean() * 100:.2f}%"
    )

    print(
        f"Std deviation: "
        f"{accuracies.std() * 100:.2f}%"
    )

    print("Chance level: 20.00%")

    return accuracies


# =========================
# WITHIN-STAGE DECODING
# =========================

acc_stim = cross_validate(
    X_stim,
    "STIMULATED DECODING"
)

acc_off1 = cross_validate(
    X_off1,
    "OFF 1 SECOND DECODING"
)

acc_off5 = cross_validate(
    X_off5,
    "OFF 5 SECOND DECODING"
)


# ============================================================
# CROSS-STAGE DECODER
# Train on stimulated brain
# Test after stimulus removal
# ============================================================

def cross_stage_test():

    print()
    print("=" * 70)
    print("CROSS-STAGE DECODING")
    print("=" * 70)

    indices = np.arange(len(labels))

    shuffled = indices.copy()
    rng.shuffle(shuffled)

    fold_indices = np.array_split(
        shuffled,
        FOLDS
    )

    results_1s = []
    results_5s = []

    for fold in range(FOLDS):

        test_idx = fold_indices[fold]

        train_idx = np.concatenate(
            [
                fold_indices[i]
                for i in range(FOLDS)
                if i != fold
            ]
        )

        train_X = X_stim[train_idx]
        train_y = labels[train_idx]

        test_1s = X_off1[test_idx]
        test_5s = X_off5[test_idx]
        test_y = labels[test_idx]

        # Select features using stimulated TRAINING data only.
        features = select_features(
            train_X,
            train_y,
            TOP_FEATURES
        )

        train_X = train_X[:, features]
        test_1s = test_1s[:, features]
        test_5s = test_5s[:, features]

        # Standardization based only on training data.
        mean = train_X.mean(axis=0)
        std = train_X.std(axis=0)

        std[std < 1e-6] = 1.0

        train_X = (
            train_X - mean
        ) / std

        test_1s = (
            test_1s - mean
        ) / std

        test_5s = (
            test_5s - mean
        ) / std

        pred_1s = classify(
            train_X,
            train_y,
            test_1s
        )

        pred_5s = classify(
            train_X,
            train_y,
            test_5s
        )

        acc1 = np.mean(
            pred_1s == test_y
        )

        acc5 = np.mean(
            pred_5s == test_y
        )

        results_1s.append(acc1)
        results_5s.append(acc5)

        print(
            f"Fold {fold + 1}: "
            f"OFF 1s = {acc1 * 100:.1f}% | "
            f"OFF 5s = {acc5 * 100:.1f}%"
        )

    print()

    print(
        f"OFF 1s cross-stage accuracy: "
        f"{np.mean(results_1s) * 100:.2f}%"
    )

    print(
        f"OFF 5s cross-stage accuracy: "
        f"{np.mean(results_5s) * 100:.2f}%"
    )

    print("Chance level: 20.00%")


cross_stage_test()


print()
print("=" * 70)
print("DECODER EXPERIMENT COMPLETE")
print("=" * 70)