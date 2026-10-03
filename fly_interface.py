import numpy as np

from flybrain import FlyBrain
from fly_semantic_state import FlySemanticState
from fly_memory import FlyMemory
from fly_internal_state import FlyInternalState
from fly_behavior import FlyBehavior


MODEL_FILE = "fly_decoder_model.npz"


class PersistentDecoder:

    def __init__(self):

        model = np.load(
            MODEL_FILE,
            allow_pickle=True
        )

        self.features = model["features"]
        self.mean = model["mean"]
        self.std = model["std"]
        self.centroids = model["centroids"]

        self.classes = [
            str(x)
            for x in model["classes"]
        ]

    def predict(self, neural_state):

        X = np.asarray(
            neural_state,
            dtype=np.float32
        )

        X = X[:, self.features]

        X = (
            X - self.mean
        ) / self.std

        X_norm = np.linalg.norm(
            X,
            axis=1,
            keepdims=True
        )

        X_norm[
            X_norm == 0
        ] = 1.0

        C_norm = np.linalg.norm(
            self.centroids,
            axis=1,
            keepdims=True
        )

        C_norm[
            C_norm == 0
        ] = 1.0

        X_normalized = X / X_norm

        C_normalized = (
            self.centroids / C_norm
        )

        similarities = (
            X_normalized
            @ C_normalized.T
        )

        index = int(
            np.argmax(
                similarities[0]
            )
        )

        confidence = float(
            similarities[0][index]
        )

        return (
            self.classes[index],
            confidence
        )


class FlyInterface:

    def __init__(self, brain=None):

        print()
        print(
            "Initializing FLY-001..."
        )

        self.brain = (
            brain
            if brain is not None
            else FlyBrain(device="auto")
        )

        self.azimuth = np.asarray(
            self.brain.azimuth
        )

        self.n_receptors = len(
            self.azimuth
        )

        self.decoder = (
            PersistentDecoder()
        )

        self.memory = FlyMemory()

        self.internal_state = (
            FlyInternalState()
        )

        self.behavior = FlyBehavior()

        self.positions = {
            "LEFT": -0.75,
            "CENTER_LEFT": -0.375,
            "CENTER": 0.0,
            "CENTER_RIGHT": 0.375,
            "RIGHT": 0.75,
        }

        print(
            f"Neurons: {self.brain.n}"
        )

        print(
            f"Visual receptors: "
            f"{self.n_receptors}"
        )

        print(
            f"Decoder classes: "
            f"{self.decoder.classes}"
        )

        print(
            f"Memory interactions: "
            f"{self.memory.get_interaction_count()}"
        )

    def encode_message(
        self,
        message
    ):

        message = (
            message
            .lower()
            .strip()
        )

        if (
            "go left" in message
            or "move left" in message
            or message == "left"
        ):

            return FlySemanticState(
                concept="MOVE",
                direction="LEFT"
            )

        if (
            "go right" in message
            or "move right" in message
            or message == "right"
        ):

            return FlySemanticState(
                concept="MOVE",
                direction="RIGHT"
            )

        if message in [
            "hello",
            "hi",
            "hey"
        ]:

            return FlySemanticState(
                concept="HELLO",
                direction="CENTER"
            )

        if message in [
            "yes",
            "yeah",
            "yep",
            "okay",
            "ok"
        ]:

            return FlySemanticState(
                concept="YES",
                direction="CENTER_RIGHT"
            )

        if message in [
            "no",
            "nope"
        ]:

            return FlySemanticState(
                concept="NO",
                direction="CENTER_LEFT"
            )

        if message in [
            "up",
            "go up"
        ]:

            return FlySemanticState(
                concept="MOVE",
                direction="CENTER"
            )

        if message in [
            "down",
            "go down"
        ]:

            return FlySemanticState(
                concept="MOVE",
                direction="CENTER"
            )

        if "move" in message:

            return FlySemanticState(
                concept="MOVE",
                direction="CENTER"
            )

        return None

    def make_stimulus(
        self,
        position,
        intensity=1.0
    ):

        direction = self.positions[
            position
        ]

        distance = np.abs(
            self.azimuth - direction
        )

        valid = np.where(
            np.isfinite(distance)
        )[0]

        selected = valid[
            np.argsort(
                distance[valid]
            )[:500]
        ]

        stimulus = np.zeros(
            self.n_receptors,
            dtype=np.float32
        )

        stimulus[selected] = intensity

        return stimulus

    def process(
        self,
        position,
        steps=50
    ):

        self.brain.reset()

        windows = []

        total_spikes = 0

        internal_drive = (
            self.internal_state
            .get_internal_drive()
        )

        print(
            f"Internal drive applied: "
            f"{internal_drive:.3f}"
        )

        for start in range(
            0,
            steps,
            10
        ):

            window_activity = np.zeros(
                self.brain.n,
                dtype=np.float32
            )

            for _ in range(10):

                stimulus = (
                    self.make_stimulus(
                        position,
                        internal_drive
                    )
                )

                fired = self.brain.step(
                    eye_drive=stimulus
                )

                total_spikes += len(
                    fired
                )

                if len(fired) > 0:

                    window_activity[
                        fired
                    ] += 1

            windows.append(
                window_activity
            )

        neural_state = np.concatenate(
            windows
        )

        decoded, confidence = (
            self.decoder.predict(
                neural_state.reshape(
                    1,
                    -1
                )
            )
        )

        return (
            decoded,
            confidence,
            total_spikes,
            internal_drive
        )

    def remember(
        self,
        message,
        state
    ):

        previous = (
            self.memory.get_current_state()
        )

        self.memory.update(
            message,
            state
        )

        return previous

    def update_internal_state(
        self,
        neural_activity
    ):

        summary = (
            self.memory.get_state_summary()
        )

        self.internal_state.update(
            summary,
            neural_activity
        )


def main():

    fly = FlyInterface()

    print()
    print("=" * 60)
    print(
        "FLY-001 SEMANTIC NEURAL INTERFACE"
    )
    print("=" * 60)

    print()
    print(
        "Talk to the simulated fly."
    )

    print(
        "Type 'memory' to inspect memory."
    )

    print(
        "Type 'recent' to inspect recent state."
    )

    print(
        "Type 'internal' to inspect internal state."
    )

    print(
        "Type 'exit' to stop."
    )

    print()

    while True:

        message = input(
            "YOU > "
        ).strip()

        if message.lower() == "exit":

            break

        if message.lower() == "memory":

            fly.memory.show_memory()

            print()

            continue

        if message.lower() == "recent":

            fly.memory.show_recent_state()

            print()

            continue

        if message.lower() == "internal":

            fly.internal_state.describe()

            print()

            continue

        state = fly.encode_message(
            message
        )

        if state is None:

            print()
            print(
                "FLY > I don't have "
                "a neural representation "
                "for that yet."
            )

            print()

            continue

        state.describe()

        print(
            "Processing through "
            "166,700-neuron brain..."
        )

        (
            decoded,
            confidence,
            total_spikes,
            applied_drive
        ) = fly.process(
            state.direction
        )

        print()
        print(
            "NEURAL DECODER"
        )

        print(
            f"Decoded direction: "
            f"{decoded}"
        )

        print(
            f"Decoder similarity: "
            f"{confidence:.4f}"
        )

        print(
            f"Total neural spikes: "
            f"{total_spikes}"
        )

        previous = fly.remember(
            message,
            state
        )

        fly.update_internal_state(
            total_spikes
        )

        print()

        fly.internal_state.describe()

        action = fly.behavior.decide(
            decoded,
            fly.internal_state
        )

        fly.behavior.evaluate(
            action,
            fly.internal_state
        )

        if previous is not None:

            print()
            print(
                "PREVIOUS FLY STATE"
            )

            print("-" * 40)

            print(
                f"Concept   : "
                f"{previous['concept']}"
            )

            print(
                f"Direction : "
                f"{previous['direction']}"
            )

            print(
                f"Intensity : "
                f"{previous['intensity']}"
            )

            print(
                f"Temporal  : "
                f"{previous['temporal']}"
            )

            print("-" * 40)

        if state.concept == "HELLO":

            if previous is None:

                response = (
                    "Hello. Neural state received."
                )

            else:

                response = (
                    "Hello. I remember my "
                    f"previous direction was "
                    f"{previous['direction']}."
                )

        elif state.concept == "YES":

            if previous is None:

                response = (
                    "Yes-state detected."
                )

            else:

                response = (
                    "Yes-state detected. "
                    f"Previous state was "
                    f"{previous['direction']}."
                )

        elif state.concept == "NO":

            if previous is None:

                response = (
                    "No-state detected."
                )

            else:

                response = (
                    "No-state detected. "
                    f"Previous state was "
                    f"{previous['direction']}."
                )

        elif state.concept == "MOVE":

            if previous is None:

                response = (
                    f"Movement state detected: "
                    f"{decoded}. "
                    f"Behavior: {action}."
                )

            else:

                response = (
                    f"Movement state detected: "
                    f"{decoded}. "
                    f"Behavior: {action}. "
                    f"Previous state was "
                    f"{previous['direction']}."
                )

        else:

            response = (
                "Neural state received."
            )

        print()
        print(
            f"FLY > {response}"
        )

        print()


if __name__ == "__main__":
    main()