import numpy as np


class FlyDecoder:

    def __init__(self, classes, top_features=1000):

        self.classes = list(classes)
        self.top_features = top_features

        self.features = None
        self.mean = None
        self.std = None
        self.centroids = None

    def _select_features(self, X, y):

        overall_mean = X.mean(axis=0)

        scores = np.zeros(
            X.shape[1],
            dtype=np.float32
        )

        for cls in self.classes:

            mask = y == cls

            if not np.any(mask):
                continue

            class_data = X[mask]

            class_mean = class_data.mean(axis=0)

            scores += len(class_data) * (
                class_mean - overall_mean
            ) ** 2

        count = min(
            self.top_features,
            X.shape[1]
        )

        features = np.argpartition(
            scores,
            -count
        )[-count:]

        return features

    def fit(self, X, y):

        X = np.asarray(
            X,
            dtype=np.float32
        )

        y = np.asarray(y)

        # Feature selection is performed
        # only on training data.
        self.features = self._select_features(
            X,
            y
        )

        X = X[:, self.features]

        self.mean = X.mean(axis=0)

        self.std = X.std(axis=0)

        self.std[self.std < 1e-6] = 1.0

        X = (
            X - self.mean
        ) / self.std

        self.centroids = []

        for cls in self.classes:

            centroid = X[
                y == cls
            ].mean(axis=0)

            self.centroids.append(
                centroid
            )

        self.centroids = np.asarray(
            self.centroids
        )

        return self

    def predict(self, X):

        X = np.asarray(
            X,
            dtype=np.float32
        )

        X = X[:, self.features]

        X = (
            X - self.mean
        ) / self.std

        # Normalize samples and centroids
        # for cosine similarity.

        X_norm = np.linalg.norm(
            X,
            axis=1,
            keepdims=True
        )

        X_norm[X_norm == 0] = 1.0

        C_norm = np.linalg.norm(
            self.centroids,
            axis=1,
            keepdims=True
        )

        C_norm[C_norm == 0] = 1.0

        X_normalized = X / X_norm

        C_normalized = (
            self.centroids / C_norm
        )

        similarities = (
            X_normalized @
            C_normalized.T
        )

        predictions = np.asarray(
            self.classes
        )[np.argmax(
            similarities,
            axis=1
        )]

        return predictions