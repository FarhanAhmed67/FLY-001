import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fly_interface import FlyInterface
from fly_2d_brain import Fly2D

print("Initializing FLY-001 web backend...")
fly = FlyInterface()
world = Fly2D()

spike_history = []
state_history = []
world_history = []
last_world_result = None
last_neural = {}


def safe_number(value):
    try:
        if isinstance(value, bool):
            return value
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (int, float)):
            return float(value)
    except Exception:
        pass
    return None


def read_attr(obj, names):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if callable(value):
                continue
            number = safe_number(value)
            if number is not None:
                return number
    return None


def world_snapshot():
    # Fly2D has changed during experimentation, so this intentionally reads
    # common state names defensively rather than modifying the validated module.
    fly_x = read_attr(world, ["fly_x", "x", "position_x"])
    fly_y = read_attr(world, ["fly_y", "y", "position_y"])
    target_x = read_attr(world, ["light_x", "target_x", "goal_x"])
    target_y = read_attr(world, ["light_y", "target_y", "goal_y"])

    # Some versions keep the world as a nested object.
    nested = getattr(world, "world", None)
    if nested is not None:
        fly_x = fly_x if fly_x is not None else read_attr(nested, ["fly_x", "x", "position_x"])
        fly_y = fly_y if fly_y is not None else read_attr(nested, ["fly_y", "y", "position_y"])
        target_x = target_x if target_x is not None else read_attr(nested, ["light_x", "target_x", "goal_x"])
        target_y = target_y if target_y is not None else read_attr(nested, ["light_y", "target_y", "goal_y"])

    direction = None
    behavior = None
    if isinstance(last_world_result, dict):
        direction = (
            last_world_result.get("decoded_direction")
            or last_world_result.get("direction")
        )
        behavior = (
            last_world_result.get("action")
            or last_world_result.get("behavior")
        )

    distance = None
    if None not in (fly_x, fly_y, target_x, target_y):
        distance = ((target_x - fly_x) ** 2 + (target_y - fly_y) ** 2) ** 0.5

    return {
        "fly": {"x": fly_x, "y": fly_y},
        "target": {"x": target_x, "y": target_y},
        "direction": direction,
        "behavior": behavior,
        "distance": distance,
        "step": len(world_history),
    }


def brain_snapshot():
    internal = fly.internal_state
    return {
        "direction": getattr(internal, "current_direction", None),
        "previous_direction": getattr(internal, "previous_direction", None),
        "internal_drive": float(internal.get_internal_drive()),
        "activity": float(getattr(internal, "neural_activity", 0.0)),
        "pattern": getattr(internal, "pattern", None),
        "stability": float(getattr(internal, "stability", 0.0)),
        "direction_changes": int(getattr(internal, "direction_changes", 0)),
        "interactions": int(fly.memory.get_interaction_count()),
        "decoded_direction": last_neural.get("decoded_direction"),
        "confidence": last_neural.get("confidence"),
        "total_spikes": last_neural.get("total_spikes"),
        "applied_drive": last_neural.get("applied_drive"),
        "concept": last_neural.get("concept"),
    }


def build_response(message):
    global spike_history, state_history, last_neural

    message = message.strip()
    if not message:
        return {"ok": False, "error": "Message cannot be empty."}

    state = fly.encode_message(message)
    if state is None:
        return {
            "ok": True,
            "supported": False,
            "response": "I don't have a neural representation for that yet.",
            "message": message,
        }

    decoded, confidence, total_spikes, applied_drive = fly.process(state.direction)
    previous = fly.remember(message, state)
    fly.update_internal_state(total_spikes)
    action = fly.behavior.decide(decoded, fly.internal_state)
    fly.behavior.evaluate(action, fly.internal_state)

    if state.concept == "HELLO":
        response = (
            "Hello. Neural state received."
            if previous is None
            else f"Hello. I remember my previous direction was {previous['direction']}."
        )
    elif state.concept == "YES":
        response = "Yes-state detected."
    elif state.concept == "NO":
        response = "No-state detected."
    elif state.concept == "MOVE":
        response = f"Movement state detected: {decoded}. Behavior: {action}."
    else:
        response = "Neural state received."

    last_neural = {
        "decoded_direction": decoded,
        "confidence": float(confidence),
        "total_spikes": int(total_spikes),
        "applied_drive": float(applied_drive),
        "concept": state.concept,
    }
    snap = brain_snapshot()
    spike_history.append(int(total_spikes))
    state_history.append({
        "direction": decoded,
        "behavior": action,
        "activity": snap["activity"],
        "drive": snap["internal_drive"],
        "confidence": float(confidence),
        "spikes": int(total_spikes),
        "concept": state.concept,
    })
    del spike_history[:-30]
    del state_history[:-30]

    return {
        "ok": True,
        "supported": True,
        "message": message,
        "response": response,
        "neural": {
            "decoded_direction": decoded,
            "confidence": float(confidence),
            "total_spikes": int(total_spikes),
            "applied_drive": float(applied_drive),
        },
        "state": {"concept": state.concept, "direction": state.direction},
        "behavior": {"action": action},
        "internal": snap,
        "memory": {"interactions": int(fly.memory.get_interaction_count()), "previous": previous},
        "history": {"spikes": spike_history, "states": state_history},
        "world": world_snapshot(),
    }


def step_world():
    global last_world_result
    result = world.run_step()
    last_world_result = result
    snap = world_snapshot()
    world_history.append(snap)
    del world_history[:-60]
    return {"ok": True, "world": snap, "raw": result if isinstance(result, (dict, list, str, int, float, bool, type(None))) else str(result)}


class Handler(BaseHTTPRequestHandler):
    def send_json(self, payload, status=200):
        data = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/status":
            self.send_json({
                "ok": True, "name": "FLY-001", "neurons": 166700,
                "visual_receptors": int(fly.n_receptors),
                "memory_interactions": int(fly.memory.get_interaction_count()),
                "decoder_classes": list(fly.decoder.classes),
            })
            return
        if self.path == "/api/telemetry":
            self.send_json({
                "ok": True,
                "brain": {"neurons": 166700, "visual_receptors": int(fly.n_receptors)},
                "internal": brain_snapshot(),
                "history": {"spikes": spike_history, "states": state_history},
            })
            return
        if self.path == "/api/world":
            self.send_json({"ok": True, "world": world_snapshot(), "history": world_history})
            return

        files = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/index.html": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "application/javascript; charset=utf-8"),
            "/style.css": ("style.css", "text/css; charset=utf-8"),
        }
        if self.path in files:
            filename, content_type = files[self.path]
            content = (WEB_ROOT / filename).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        # Serve only the bundled 3D model from the web/assets directory.
        # No arbitrary filesystem paths are accepted.
        if self.path == "/assets/fruit-fly.glb":
            asset = WEB_ROOT / "assets" / "fruit-fly.glb"
            if asset.is_file():
                content = asset.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "model/gltf-binary")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                self.wfile.write(content)
                return

        self.send_json({"ok": False, "error": "Not found."}, 404)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            if self.path == "/api/chat":
                self.send_json(build_response(body.get("message", "")))
                return
            if self.path == "/api/world/step":
                self.send_json(step_world())
                return
            self.send_json({"ok": False, "error": "Not found."}, 404)
        except Exception as exc:
            self.send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)

    def log_message(self, fmt, *args):
        print("[WEB]", fmt % args)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("\n" + "=" * 58)
    print("FLY-001 WEB INTERFACE v0.8")
    print("=" * 58)
    print("Open:      http://127.0.0.1:8000")
    print("API:       http://127.0.0.1:8000/api/status")
    print("Telemetry: http://127.0.0.1:8000/api/telemetry")
    print("World:     http://127.0.0.1:8000/api/world")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping FLY-001 web interface...")
    finally:
        server.server_close()
