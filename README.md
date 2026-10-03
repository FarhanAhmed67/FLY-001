 FLY-001

 A Computational Fruit-Fly Nervous System Simulation

FLY-001 is an experimental computational system built around a simulated fruit-fly nervous system containing approximately **166,700 neurons** and **6,006 visual receptors**.

The project explores how a large neural system can transform visual input into persistent neural activity, internal state, memory-like representations, behavioral decisions, and closed-loop interaction with a simulated environment.

The system is not intended to claim biological consciousness or genuine semantic understanding. It is an experimental computational model for studying neural representations and behavior.

---

 Overview

FLY-001 combines a simulated neural system with perception, decoding, internal state, memory, behavior, and a 2D environment.

```text
                         FLY-001
                            │
             ┌──────────────┴──────────────┐
             │                             │
        WORLD INPUT                   USER INPUT
             │                             │
          Vision                    Semantic Encoder
             │                             │
             └──────────────┬──────────────┘
                            │
                         FLYBRAIN
                            │
                    Neural Processing
                            │
                         Decoder
                            │
             ┌──────────────┴──────────────┐
             │                             │
       Internal State                    Memory
             │                             │
             └──────────────┬──────────────┘
                            │
                         Behavior
                            │
                           World
                            │
                         Feedback
                            └───────────────►

```

Core System
Neural System
FLY-001 uses the FlyBrain computational model with:
- 166,700 neurons
- 6,006 visual receptors
- CPU-based neural simulation
- Visual stimulation through neural inputs
- Motor/neural activity monitoring
- Neural state reset and repeated simulation steps
The neural system is exposed through a Python interface that allows visual inputs to be converted into neural activity.
Architecture
The project is composed of several major components.
Component	Purpose
fly_core.py	Core FLY-001 neural interface
fly_interface.py	User-facing interaction with the neural system
fly_2d_brain.py	Neural processing for the 2D environment
fly_brain_world.py	Closed-loop brain/world integration
fly_vision.py	Visual input processing
fly_world.py	World/environment representation
fly_behavior.py	Behavioral decision layer
fly_internal_state.py	Internal state representation
fly_memory.py	Memory/state persistence
fly_encoder.py	Input encoding
fly_decoder.py	Neural output decoding
semantic_encoder.py	Semantic input encoding
web/	Web interface and API

```
Neural Processing
The core neural loop follows the pattern:
Visual Input
     │
     ▼
Visual Receptors
     │
     ▼
FlyBrain
     │
     ▼
Neural Spikes
     │
     ▼
Motor / Population Decoders
     │
     ▼
Behavior
```
The system can monitor neural firing and aggregate activity across defined motor groups.
Example motor groups include:
- Forward
- Steering
- Escape
- Backward
- Punch
- Kick
Experiments
A major part of FLY-001 is experimental validation of neural representations and behavior.
The repository contains experiments covering:
- Spatial decoding
- Position representation
- Intensity representation
- Temporal representation
- Context generalization
- Compositional representations
- Neural persistence
- Semantic decoding
- Encoder/decoder validation
- Closed-loop control
- Navigation
- Obstacle avoidance
- Moving-target pursuit
- Predictive behavior
Representation Experiments
Spatial Representation
A five-position visual experiment produced population similarity around:
0.9725 – 0.9758
A strict held-out context experiment achieved:
100% accuracy
against a 20% chance baseline.
Intensity Representation
Fixed-context 5-fold cross-validation achieved:
73.75% accuracy
against a 25% chance baseline.
However, strict held-out context intensity generalization achieved:
33.33%
which corresponds to the 33.33% chance baseline.
This suggests that intensity information can be decoded under familiar conditions but does not necessarily generalize across unseen contexts.
Temporal Representation
Fixed-context 5-fold cross-validation achieved:
100% accuracy
against a 25% chance baseline.
Strict held-out temporal context achieved:
22%
against a 25% chance baseline.
Compositionality
A full 3D compositional experiment produced:
0.67% accuracy
against a 1.67% chance baseline.
Individual factor decoding produced approximately:
Factor	Accuracy	Chance
Position	18.33%	20%
Intensity	32.00%	33.33%
Temporal	25.67%	25%


These results indicate that the tested representation did not demonstrate strong independent compositional decoding across the full combination space.
Persistence
After visual stimulation was removed, elevated ON/OFF activity could remain for at least 5 seconds.
This demonstrates persistent neural activity in the tested system.
It should not be interpreted as biological memory, consciousness, or subjective experience.
Semantic Experiment
A semantic decoder achieved:
67.5% accuracy
against a 10% chance baseline in the tested experiment.
However, the semantic labels were externally defined and mapped onto neural activity.
Therefore, this result should not be interpreted as evidence that the simulated nervous system possesses genuine semantic understanding.
Closed-Loop Control
FLY-001 can connect its neural processing directly to a simulated environment.

```
Environment
     │
     ▼
Visual Encoder
     │
     ▼
FLY-001 Brain
     │
     ▼
Decoder
     │
     ▼
Movement
     │
     ▼
Environment
     │
     └────────── Feedback
```
A five-position closed-loop encoder → brain → decoder experiment achieved:
100% classification accuracy
with labels supplied externally.
2D Navigation
FLY-001 was tested in a simulated 2D environment.
Stationary Target Navigation
Benchmark:
20 / 20 successful trials
100% success rate
Average distance reduction:
96.91%
Mean final distance:
1.09
Obstacle Avoidance
A stationary obstacle benchmark produced:
20 / 20 successful trials
100% success rate
Average distance reduction:
97.99%
Mean final distance:
1.51
Mean number of steps:
22.40
Moving Target + Obstacle Navigation
The system was also tested against a moving target while avoiding an obstacle.
Configuration:
Trials:                 20
Maximum steps:          100
Target threshold:       2
Target speed:           3
Prediction horizon:     2
Neural update interval: 5
Obstacle margin:        4
Random seed:             42

Neurons:                166,700
Visual receptors:         6,006

Results:
Metric	Result
Successful trials	20 / 20
Success rate	100%
Mean distance reduction	99.12%
Median distance reduction	99.38%
Mean final distance	0.70
Median final distance	0.51
Best final distance	0
Worst final distance	1.96


Route diagnostics
The direct path was blocked in:
15 / 20 trials
Route decisions for blocked paths:
Above: 23
Below: 15
No route: 0

The benchmark demonstrates successful closed-loop navigation under the tested simulation conditions.
Predictive Pursuit
A 1D moving-target experiment was also performed.
Pure pursuit achieved:
6 / 20 successful trials
while a predictive horizon of 2 achieved:
17 / 20 successful trials
or:
85% success
in the tested benchmark.
This experiment suggests that predictive target estimation can significantly improve pursuit performance within the simulation.
Web Interface
FLY-001 includes a browser-based interface for interacting with the system.
The interface provides:
- Chat interaction
- Neural telemetry
- Brain activity visualization
- Internal state information
- 2D world visualization
- 3D fruit-fly model
- Session-based backend
- Closed-loop world interaction
The interface is divided into:
CHAT
BRAIN
WORLD


```
The deployed interface uses a black-and-white terminal-inspired design.
Web Architecture
Browser
   │
   ▼
Nginx / HTTPS
   │
   ▼
Python Web Server
   │
   ├── Session Manager
   │
   ├── Chat API
   │
   ├── Telemetry API
   │
   └── World API
          │
          ▼
      FLY-001
          │
          ▼
      FlyBrain

Each browser session receives an isolated FLY-001 state.
The deployment also limits simultaneous sessions because the neural model is computationally and memory intensive.
Repository Structure
FLY-001/
│
├── fly_core.py
├── fly_interface.py
├── fly_2d_brain.py
├── fly_brain_world.py
├── fly_behavior.py
├── fly_internal_state.py
├── fly_memory.py
├── fly_vision.py
├── fly_world.py
│
├── fly_encoder.py
├── fly_decoder.py
├── semantic_encoder.py
│
├── Experiments/
│   ├── spatial_test.py
│   ├── temporal_test.py
│   ├── intensity_test.py
│   ├── compositional_test.py
│   └── ...
│
├── web/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── assets/
│       ├── fruit-fly.glb
│       └── fly001-logo-white-transparent.png
│
├── requirements.txt
└── .gitignore
```
Requirements
- Python 3.12+
- NumPy
- SciPy
- scikit-learn
- Numba
- FlyBrain
- Matplotlib
- Pillow
Install dependencies:
pip install -r requirements.txt

Brain Data
The neural simulation requires the FlyBrain data files:
brain.npz
weights.npz

These files are intentionally not included in this repository because of their size.
Set the data directory using:
export FLY_DATA=/path/to/fly-data

On Windows PowerShell:
$env:FLY_DATA="C:\path\to\fly-data"

For CPU execution:
export FLY_DEVICE=cpu

Running FLY-001
After installing the requirements and providing the required neural data:
python fly_core.py

The web interface can be started through:
python web/server.py

The default development server runs on:
http://127.0.0.1:8000

Current Limitations
FLY-001 is an experimental computational system.
The following limitations are important:
- The simulated nervous system is not a complete biological fruit fly.
- Neural activity should not be interpreted as consciousness.
- Persistent neural activity is not equivalent to biological memory.
- Semantic decoding does not demonstrate genuine semantic understanding.
- Several experiments show performance close to chance under strict held-out conditions.
- Behavioral success depends on the defined environment, encoders, decoders, and benchmark conditions.
- The system currently operates primarily through CPU-based simulation.
- The large neural model requires significant memory.
Results reported in this README correspond to the specific experimental configurations used during development.
Research Direction
FLY-001 is being developed as an experimental platform for exploring:
- Neural representation
- Emergent behavior
- Closed-loop neural control
- Visual processing
- Neural persistence
- Memory-like state
- Predictive behavior
- Brain-to-behavior decoding
- Artificial nervous systems
The long-term goal is to explore how complex behavior can emerge from interactions between neural representations, internal state, memory, perception, and an environment.
Disclaimer
FLY-001 is a computational research project.
It does not claim to reproduce the full biological nervous system of Drosophila melanogaster, nor does it claim that the simulated system is conscious, sentient, or capable of genuine human-like understanding.
Experimental results should be interpreted within the specific simulation and benchmark conditions under which they were obtained.
Author
Farhan Ahmed
GitHub:
https://github.com/FarhanAhmed67
