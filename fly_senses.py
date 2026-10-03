import numpy as np
from fly_core import Fly001
from fly_state import FlyState


class FlySenses:

    def __init__(self, fly):
        self.fly = fly
        self.n = fly.n_visual

    def dark(self):
        return np.zeros(self.n, dtype=np.float32)

    def bright(self):
        return np.ones(self.n, dtype=np.float32)

    def dim(self, intensity=0.25):
        intensity = np.clip(intensity, 0.0, 1.0)
        return np.full(self.n, intensity, dtype=np.float32)

    def first_half_light(self, intensity=1.0):
        intensity = np.clip(intensity, 0.0, 1.0)
        visual = np.zeros(self.n, dtype=np.float32)
        visual[:self.n // 2] = intensity
        return visual

    def second_half_light(self, intensity=1.0):
        intensity = np.clip(intensity, 0.0, 1.0)
        visual = np.zeros(self.n, dtype=np.float32)
        visual[self.n // 2:] = intensity
        return visual

    def center_light(self, width=0.2, intensity=1.0):
        width = np.clip(width, 0.01, 1.0)
        intensity = np.clip(intensity, 0.0, 1.0)
        visual = np.zeros(self.n, dtype=np.float32)
        center = self.n // 2
        half_width = int((self.n * width) / 2)
        start = max(0, center - half_width)
        end = min(self.n, center + half_width)
        visual[start:end] = intensity
        return visual


if __name__ == "__main__":

    fly = Fly001()
    senses = FlySenses(fly)
    state = FlyState(fly)

    print("FLY-001 sensory system initialized")
    print("Visual receptors:", senses.n)

    scenes = {
        "DARK": senses.dark(),
        "DIM": senses.dim(0.25),
        "BRIGHT": senses.bright(),
        "FIRST HALF LIGHT": senses.first_half_light(),
        "SECOND HALF LIGHT": senses.second_half_light(),
        "CENTER LIGHT": senses.center_light()
    }

    for name, visual in scenes.items():

        fly.reset()

        result = state.run(visual, steps=20)

        print(f"\\n========== {name} ==========")
        print("Visual:", f"mean={result['visual_mean']:.2f}", f"min={result['visual_min']:.2f}", f"max={result['visual_max']:.2f}")
        print("Total neural spikes:", result["total_spikes"])
        print("Movement:", result["movement"])
        print("Steering:", result["steering"])
        print("Escape:", result["escape"])
        print("Other:", result["other"])
