"""
FLY-001 Master Controller

This file is an orchestration layer only.
It does not modify the validated neural experiments or their modules.

Available modes:
  1 = 2D brain-controlled world
  2 = brain -> behavior -> world
  3 = semantic neural interface
  4 = system status
  5 = validated moving-target + obstacle benchmark
  0 = exit
"""

import subprocess
import sys

from fly_2d_brain import Fly2D
from fly_brain_world import ClosedLoopWorld
from fly_interface import FlyInterface


class FLY001:
    """Master controller for the existing FLY-001 subsystems."""

    def __init__(self):
        self.name = "FLY-001"
        self.version = "0.2"

    def status(self):
        print()
        print("=" * 60)
        print("FLY-001 SYSTEM STATUS")
        print("=" * 60)
        print("Core brain         : FlyBrain / 166,700 neurons")
        print("Visual receptors   : 6,006")
        print("2D world           : READY")
        print("Closed-loop world  : READY")
        print("Semantic interface : READY")
        print("Navigation benchmark: READY")
        print("Memory             : ENABLED")
        print("Internal state     : ENABLED")
        print("Behavior system    : ENABLED")
        print("Validated tests    : PRESERVED")
        print("=" * 60)

    def run_2d(self, steps=10):
        """Run the existing Fly2D closed-loop demonstration."""
        simulation = Fly2D()

        print()
        print("=" * 60)
        print("FLY-001 MASTER → 2D BRAIN CONTROL")
        print("=" * 60)
        print("Fly starts at (50,50)")
        print("Light starts at (80,50)")
        print(f"Running {steps} brain-controlled steps...")

        for _ in range(steps):
            simulation.run_step()

    def run_world(self, steps=10):
        """Run the existing brain -> behavior -> world demonstration."""
        simulation = ClosedLoopWorld()

        print()
        print("=" * 60)
        print("FLY-001 MASTER → BRAIN → BEHAVIOR → WORLD")
        print("=" * 60)
        print("Light starts at X=80")
        print("Fly starts at X=50")
        print(f"Running {steps} closed-loop steps...")

        for _ in range(steps):
            simulation.run_step()

    def run_interface(self):
        """Run the existing semantic neural interface."""
        fly = FlyInterface()

        print()
        print("=" * 60)
        print("FLY-001 MASTER → SEMANTIC NEURAL INTERFACE")
        print("=" * 60)
        print("Talk to the simulated fly.")
        print("Type 'memory' to inspect memory.")
        print("Type 'recent' to inspect recent state.")
        print("Type 'internal' to inspect internal state.")
        print("Type 'exit' to return to the master menu.")

        while True:
            message = input("\nYOU > ").strip()

            if message.lower() == "exit":
                break

            if message.lower() == "memory":
                fly.memory.show_memory()
                continue

            if message.lower() == "recent":
                fly.memory.show_recent_state()
                continue

            if message.lower() == "internal":
                fly.internal_state.describe()
                continue

            state = fly.encode_message(message)

            if state is None:
                print("\nFLY > I don't have a neural representation for that yet.")
                continue

            state.describe()
            print("Processing through 166,700-neuron brain...")

            decoded, confidence, total_spikes, applied_drive = fly.process(
                state.direction
            )

            print()
            print("NEURAL DECODER")
            print("----------------------------------------")
            print(f"Decoded direction : {decoded}")
            print(f"Decoder similarity: {confidence:.4f}")
            print(f"Total neural spikes: {total_spikes}")
            print(f"Internal drive applied: {applied_drive:.3f}")

            previous = fly.remember(message, state)
            fly.update_internal_state(total_spikes)

            print()
            print("FLY INTERNAL STATE")
            print("----------------------------------------")
            fly.internal_state.describe()

            action = fly.behavior.decide(
                decoded,
                fly.internal_state
            )

            print()
            print("FLY BEHAVIOR")
            print("----------------------------------------")
            print(f"Action : {action}")

            if previous is not None:
                print()
                print("PREVIOUS FLY STATE")
                print("----------------------------------------")
                print(f"Concept   : {previous['concept']}")
                print(f"Direction : {previous['direction']}")
                print(f"Intensity : {previous['intensity']}")
                print(f"Temporal  : {previous['temporal']}")

            if state.concept == "HELLO":
                if previous is None:
                    response = "Hello. Neural state received."
                else:
                    response = (
                        "Hello. I remember my previous direction was "
                        f"{previous['direction']}."
                    )
            elif state.concept == "YES":
                if previous is None:
                    response = "Yes-state detected."
                else:
                    response = (
                        "Yes-state detected. Previous state was "
                        f"{previous['direction']}."
                    )
            elif state.concept == "NO":
                if previous is None:
                    response = "No-state detected."
                else:
                    response = (
                        "No-state detected. Previous state was "
                        f"{previous['direction']}."
                    )
            elif state.concept == "MOVE":
                if previous is None:
                    response = (
                        f"Movement state detected: {decoded}. "
                        f"Behavior: {action}."
                    )
                else:
                    response = (
                        f"Movement state detected: {decoded}. "
                        f"Behavior: {action}. Previous state was "
                        f"{previous['direction']}."
                    )
            else:
                response = "Neural state received."

            print()
            print(f"FLY > {response}")

    def run_navigation_benchmark(self):
        """
        Run the existing validated moving-target + obstacle benchmark
        as a separate process.

        The benchmark file itself is intentionally untouched. Running it
        this way preserves its validated configuration and results.
        """
        benchmark = "fly_2d_moving_obstacle_benchmark.py"

        print()
        print("=" * 60)
        print("FLY-001 MASTER → MOVING TARGET + OBSTACLE")
        print("=" * 60)
        print("Launching validated benchmark:")
        print(f"  {benchmark}")
        print()
        print("The validated benchmark is being run unchanged.")
        print("=" * 60)

        try:
            result = subprocess.run(
                [sys.executable, benchmark],
                check=False
            )

            print()
            print("=" * 60)

            if result.returncode == 0:
                print("NAVIGATION BENCHMARK FINISHED")
                print("Exit code: 0")
            else:
                print("NAVIGATION BENCHMARK FINISHED WITH ERROR")
                print(f"Exit code: {result.returncode}")

            print("=" * 60)

        except FileNotFoundError:
            print()
            print("ERROR: Could not find")
            print(f"  {benchmark}")
            print("Make sure it is in the FLY-001 project folder.")

    def menu(self):
        while True:
            print()
            print("=" * 60)
            print("FLY-001 MASTER CONTROLLER")
            print("=" * 60)
            print("1. 2D brain-controlled world")
            print("2. Brain → behavior → world")
            print("3. Semantic neural interface")
            print("4. System status")
            print("5. Moving target + obstacle benchmark")
            print("0. Exit")
            print("=" * 60)

            choice = input("Select mode > ").strip()

            if choice == "1":
                self.run_2d()
            elif choice == "2":
                self.run_world()
            elif choice == "3":
                self.run_interface()
            elif choice == "4":
                self.status()
            elif choice == "5":
                self.run_navigation_benchmark()
            elif choice == "0":
                print("\nFLY-001 shutdown.")
                break
            else:
                print("\nInvalid choice. Select 0, 1, 2, 3, 4, 5, or 0.")


if __name__ == "__main__":
    FLY001().menu()
