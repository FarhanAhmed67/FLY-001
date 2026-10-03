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
