import json
import os
import secrets
import sys
import threading
import time
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flybrain import FlyBrain
from fly_interface import FlyInterface
from fly_2d_brain import Fly2D

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
SESSION_COOKIE = "fly001_session"
SESSION_TTL_SECONDS = 60 * 60
MAX_SESSIONS = 1

sessions = {}
sessions_lock = threading.Lock()


class Session:
    def __init__(self, session_id):
        self.id = session_id
        self.created_at = time.time()
        self.last_seen = self.created_at
        self.lock = threading.RLock()

        # One neural core is shared by chat and world.
        # This avoids loading two 166,700-neuron FlyBrain instances.
        self.brain = FlyBrain(device="auto")
        self.fly = FlyInterface(brain=self.brain)
        self.world = Fly2D(brain=self.brain)

        self.spike_history = []
        self.state_history = []
        self.world_history = []
        self.last_world_result = None

    def touch(self):
        self.last_seen = time.time()


def cleanup_sessions_locked():
    now = time.time()
    expired = [
        sid for sid, session in sessions.items()
        if now - session.last_seen > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        del sessions[sid]


def create_session():
    with sessions_lock:
        cleanup_sessions_locked()

        if len(sessions) >= MAX_SESSIONS:
            raise RuntimeError("FLY-001 is currently at session capacity.")

        session_id = secrets.token_urlsafe(32)
        session = Session(session_id)
        sessions[session_id] = session

    return session


def get_session(session_id):
    if not session_id:
        return None

    with sessions_lock:
        session = sessions.get(session_id)

    if session is not None:
        session.touch()

    return session


def get_or_create_session(handler):
    raw = handler.headers.get("Cookie", "")
    jar = cookies.SimpleCookie()

    try:
        jar.load(raw)
    except cookies.CookieError:
        jar = cookies.SimpleCookie()

    session_id = jar[SESSION_COOKIE].value if SESSION_COOKIE in jar else None
    session = get_session(session_id)

    if session is not None:
        return session

    try:
        session = create_session()
    except RuntimeError as exc:
        handler.session_capacity_error = str(exc)
        return None

    handler.new_session_id = session.id
    return session


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


def world_snapshot(session):
    world = session.world

    fly_x = read_attr(world, ["fly_x", "x", "position_x"])
    fly_y = read_attr(world, ["fly_y", "y", "position_y"])
    target_x = read_attr(world, ["light_x", "target_x", "goal_x"])
    target_y = read_attr(world, ["light_y", "target_y", "goal_y"])

    nested = getattr(world, "world", None)

    if nested is not None:
        fly_x = (
            fly_x
            if fly_x is not None
            else read_attr(nested, ["fly_x", "x", "position_x"])
        )

        fly_y = (
            fly_y
            if fly_y is not None
            else read_attr(nested, ["fly_y", "y", "position_y"])
        )

        target_x = (
            target_x
            if target_x is not None
            else read_attr(nested, ["light_x", "target_x", "goal_x"])
        )

        target_y = (
            target_y
            if target_y is not None
            else read_attr(nested, ["light_y", "target_y", "goal_y"])
        )

    direction = None
    behavior = None

    if isinstance(session.last_world_result, dict):
        direction = (
            session.last_world_result.get("decoded_direction")
            or session.last_world_result.get("direction")
        )

        behavior = (
            session.last_world_result.get("action")
            or session.last_world_result.get("behavior")
        )

    distance = None

    if None not in (fly_x, fly_y, target_x, target_y):
        distance = (
            (target_x - fly_x) ** 2
            + (target_y - fly_y) ** 2
        ) ** 0.5

    return {
        "fly": {"x": fly_x, "y": fly_y},
        "target": {"x": target_x, "y": target_y},
        "direction": direction,
        "behavior": behavior,
        "distance": distance,
        "step": len(session.world_history),
    }


def brain_snapshot(session):
    fly = session.fly
    internal = fly.internal_state

    return {
        "direction": getattr(internal, "current_direction", None),
        "previous_direction": getattr(
            internal,
            "previous_direction",
            None,
        ),
        "internal_drive": float(
            internal.get_internal_drive()
        ),
        "activity": float(
            getattr(internal, "neural_activity", 0.0)
        ),
        "pattern": getattr(internal, "pattern", None),
        "stability": float(
            getattr(internal, "stability", 0.0)
        ),
        "direction_changes": int(
            getattr(internal, "direction_changes", 0)
        ),
        "interactions": int(
            fly.memory.get_interaction_count()
        ),
    }


def build_response(session, message):
    message = message.strip()

    if not message:
        return {
            "ok": False,
            "error": "Message cannot be empty.",
        }

    fly = session.fly
    state = fly.encode_message(message)

    if state is None:
        return {
            "ok": True,
            "supported": False,
            "response": (
                "I don't have a neural representation "
                "for that yet."
            ),
            "message": message,
            "session": {"id": session.id},
        }

    decoded, confidence, total_spikes, applied_drive = fly.process(
        state.direction
    )

    previous = fly.remember(message, state)
    fly.update_internal_state(total_spikes)

    action = fly.behavior.decide(
        decoded,
        fly.internal_state,
    )

    fly.behavior.evaluate(
        action,
        fly.internal_state,
    )

    if state.concept == "HELLO":
        response = (
            "Hello. Neural state received."
            if previous is None
            else (
                "Hello. I remember my previous "
                f"direction was {previous['direction']}."
            )
        )

    elif state.concept == "YES":
        response = "Yes-state detected."

    elif state.concept == "NO":
        response = "No-state detected."

    elif state.concept == "MOVE":
        response = (
            f"Movement state detected: {decoded}. "
            f"Behavior: {action}."
        )

    else:
        response = "Neural state received."

    snap = brain_snapshot(session)

    session.spike_history.append(
        int(total_spikes)
    )

    session.state_history.append({
        "direction": decoded,
        "behavior": action,
        "activity": snap["activity"],
        "drive": snap["internal_drive"],
    })

    del session.spike_history[:-30]
    del session.state_history[:-30]

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
        "state": {
            "concept": state.concept,
            "direction": state.direction,
        },
        "behavior": {
            "action": action,
        },
        "internal": snap,
        "memory": {
            "interactions": int(
                fly.memory.get_interaction_count()
            ),
            "previous": previous,
        },
        "history": {
            "spikes": session.spike_history,
            "states": session.state_history,
        },
        "world": world_snapshot(session),
        "session": {"id": session.id},
    }


def step_world(session):
    result = session.world.run_step()

    session.last_world_result = result

    snap = world_snapshot(session)

    session.world_history.append(snap)
    del session.world_history[:-60]

    return {
        "ok": True,
        "world": snap,
        "raw": (
            result
            if isinstance(
                result,
                (
                    dict,
                    list,
                    str,
                    int,
                    float,
                    bool,
                    type(None),
                ),
            )
            else str(result)
        ),
        "session": {"id": session.id},
    }


class Handler(BaseHTTPRequestHandler):
    new_session_id = None

    def send_json(self, payload, status=200):
        data = json.dumps(
            payload,
            default=str,
        ).encode("utf-8")

        self.send_response(status)

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )

        self.send_header(
            "Content-Length",
            str(len(data)),
        )

        self.send_header(
            "Cache-Control",
            "no-store",
        )

        self.send_header(
            "Access-Control-Allow-Origin",
            "*",
        )

        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type",
        )

        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS",
        )

        if self.new_session_id:
            self.send_header(
                "Set-Cookie",
                (
                    f"{SESSION_COOKIE}="
                    f"{self.new_session_id}; "
                    "Path=/; HttpOnly; SameSite=Lax"
                ),
            )

            self.new_session_id = None

        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)

        self.send_header(
            "Access-Control-Allow-Origin",
            "*",
        )

        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type",
        )

        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS",
        )

        self.end_headers()

    def require_session(self):
        session = get_or_create_session(self)

        if session is None:
            self.send_json({
                "ok": False,
                "error": getattr(
                    self,
                    "session_capacity_error",
                    "Unable to create session.",
                ),
                "code": "SESSION_CAPACITY",
            }, status=503)
            return None

        return session

    def do_GET(self):
        if self.path == "/api/status":
            session = self.require_session()
            if session is None:
                return
            fly = session.fly

            self.send_json({
                "ok": True,
                "name": "FLY-001",
                "version": "1.0-session",
                "neurons": 166700,
                "visual_receptors": int(
                    fly.n_receptors
                ),
                "memory_interactions": int(
                    fly.memory.get_interaction_count()
                ),
                "decoder_classes": list(
                    fly.decoder.classes
                ),
                "session": {
                    "id": session.id,
                    "created_at": session.created_at,
                    "last_seen": session.last_seen,
                },
            })

            return

        if self.path == "/api/session":
            session = self.require_session()
            if session is None:
                return

            self.send_json({
                "ok": True,
                "session": {
                    "id": session.id,
                    "created_at": session.created_at,
                    "last_seen": session.last_seen,
                    "memory_interactions": int(
                        session.fly.memory.get_interaction_count()
                    ),
                },
            })

            return

        if self.path == "/api/telemetry":
            session = self.require_session()
            if session is None:
                return

            self.send_json({
                "ok": True,
                "brain": {
                    "neurons": 166700,
                    "visual_receptors": int(
                        session.fly.n_receptors
                    ),
                },
                "internal": brain_snapshot(session),
                "history": {
                    "spikes": session.spike_history,
                    "states": session.state_history,
                },
            })

            return

        if self.path == "/api/world":
            session = self.require_session()
            if session is None:
                return

            self.send_json({
                "ok": True,
                "world": world_snapshot(session),
                "history": session.world_history,
            })

            return

        if self.path == "/assets/fruit-fly.glb":
            asset = (
                WEB_ROOT
                / "assets"
                / "fruit-fly.glb"
            )

            if asset.is_file():
                content = asset.read_bytes()

                self.send_response(200)

                self.send_header(
                    "Content-Type",
                    "model/gltf-binary",
                )

                self.send_header(
                    "Content-Length",
                    str(len(content)),
                )

                self.send_header(
                    "Cache-Control",
                    "public, max-age=3600",
                )

                self.end_headers()
                self.wfile.write(content)

                return

        files = {
            "/": (
                "index.html",
                "text/html; charset=utf-8",
            ),
            "/index.html": (
                "index.html",
                "text/html; charset=utf-8",
            ),
            "/app.js": (
                "app.js",
                "application/javascript; charset=utf-8",
            ),
            "/style.css": (
                "style.css",
                "text/css; charset=utf-8",
            ),
        }

        if self.path in files:
            filename, content_type = files[self.path]

            content = (
                WEB_ROOT / filename
            ).read_bytes()

            self.send_response(200)

            self.send_header(
                "Content-Type",
                content_type,
            )

            self.send_header(
                "Content-Length",
                str(len(content)),
            )

            self.end_headers()
            self.wfile.write(content)

            return

        self.send_json(
            {
                "ok": False,
                "error": "Not found.",
            },
            404,
        )

    def do_POST(self):
        try:
            length = int(
                self.headers.get(
                    "Content-Length",
                    "0",
                )
            )

            body = (
                json.loads(
                    self.rfile.read(length)
                    .decode("utf-8")
                )
                if length
                else {}
            )

            session = self.require_session()
            if session is None:
                return

            with session.lock:
                if self.path == "/api/chat":
                    self.send_json(
                        build_response(
                            session,
                            body.get(
                                "message",
                                "",
                            ),
                        )
                    )
                    return

                if self.path == "/api/world/step":
                    self.send_json(
                        step_world(session)
                    )
                    return

            self.send_json(
                {
                    "ok": False,
                    "error": "Not found.",
                },
                404,
            )

        except Exception as exc:
            self.send_json(
                {
                    "ok": False,
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                },
                500,
            )

    def log_message(self, fmt, *args):
        print("[WEB]", fmt % args)


if __name__ == "__main__":
    print(
        "Initializing FLY-001 web backend..."
    )

    server = ThreadingHTTPServer(
        (HOST, PORT),
        Handler,
    )

    print("\n" + "=" * 58)
    print(
        "FLY-001 WEB INTERFACE v1.0 â€” "
        "SESSION BACKEND"
    )
    print("=" * 58)
    print(
        "Open:      "
        f"http://{HOST}:{PORT}"
    )
    print(
        "API:       "
        f"http://{HOST}:{PORT}/api/status"
    )
    print(
        "Session:   "
        f"http://{HOST}:{PORT}/api/session"
    )
    print(
        "Telemetry: "
        f"http://{HOST}:{PORT}/api/telemetry"
    )
    print(
        "World:     "
        f"http://{HOST}:{PORT}/api/world"
    )
    print(
        "Sessions:  "
        f"max {MAX_SESSIONS}, "
        f"idle timeout "
        f"{SESSION_TTL_SECONDS // 60} min"
    )
    print("Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(
            "\nStopping FLY-001 web interface..."
        )
    finally:
        server.server_close()



