"""Run Fridge Guardian on a laptop, without the UNO Q.

Uses the exact same fridge_logic.py and web page as the board. Only the
camera, sensors and outputs are simulated. Standard library only.

    python sim/run_sim.py                # open http://localhost:8000
    python sim/run_sim.py --speed 3600   # 1 real second = 1 hour in the fridge
    python sim/run_sim.py --demo         # pre-fill with a few items of different ages
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app")
sys.path.insert(0, os.path.join(APP, "python"))

from fridge_logic import FridgeGuardian  # noqa: E402

SIM_LABELS = ["milk", "cheese", "egg", "cream", "chicken", "shrimp", "strawberry",
              "tomato", "leafy_greens", "banana", "apple", "butter"]


class SimClock:
    """Fridge time that can run faster than real time."""

    def __init__(self, speed: float):
        self.speed = speed
        self.t0_real = time.time()
        self.t0_sim = time.time()

    def __call__(self) -> float:
        return self.t0_sim + (time.time() - self.t0_real) * self.speed


class FakeCamera:
    """Pretends the camera sees a label for a few frames, like a real detection."""

    def __init__(self, fridge: FridgeGuardian, webcam: "Webcam", fps: float = 5.0):
        self.fridge = fridge
        self.webcam = webcam
        self.fps = fps
        self.queue: list[str] = []
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def show(self, label: str, frames: int = 4) -> None:
        with self.lock:
            self.queue.extend([label] * frames)

    def _run(self) -> None:
        while True:
            with self.lock:
                label = self.queue.pop(0) if self.queue else None
            if label:
                self.fridge.on_detections({label: [{"confidence": 0.86, "bounding_box_xyxy": (40, 40, 200, 220)}]})
            elif not self.webcam.active():  # empty frames, unless the real webcam is feeding us
                self.fridge.on_detections({})
            time.sleep(1.0 / self.fps)


class Webcam:
    """Latest annotated frame posted by sim/webcam_detector.py."""

    def __init__(self):
        self.jpeg: bytes | None = None
        self.ts = 0.0

    def active(self) -> bool:
        return self.jpeg is not None and time.time() - self.ts < 3


def make_handler(fridge: FridgeGuardian, camera: FakeCamera, speed: float, webcam: Webcam):
    assets = os.path.join(APP, "assets")

    def state() -> dict:
        s = fridge.snapshot()
        s["sim"] = {"speed": speed, "labels": SIM_LABELS, "webcam": webcam.active()}
        return s

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the console quiet
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/api/state":
                return self._json(state())
            if url.path == "/api/sim/frame.jpg":
                if not webcam.jpeg:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(webcam.jpeg)))
                self.end_headers()
                self.wfile.write(webcam.jpeg)
                return
            path = "index.html" if url.path in ("/", "") else url.path.lstrip("/")
            full = os.path.normpath(os.path.join(assets, path))
            if not full.startswith(assets) or not os.path.isfile(full):
                self.send_error(404)
                return
            with open(full, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(full)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            route = url.path.removeprefix("/api/")
            ok = True
            if route == "add":
                ok = fridge.add_item(q.get("label", ""), float(q["shelf_days"]) if q.get("shelf_days") else None) is not None
            elif route == "remove":
                ok = fridge.remove_item(q.get("item_id", ""))
            elif route == "shelf":
                ok = fridge.set_shelf_days(q.get("item_id", ""), float(q.get("days", 1)))
            elif route == "mode":
                fridge.set_mode(q.get("mode", "in"))
            elif route == "clear":
                fridge.clear()
            elif route == "silence":
                fridge.silence()
            elif route == "sim/show":
                camera.show(q.get("label", "banana"))
            elif route == "sim/detections":
                # Posted by sim/webcam_detector.py: JSON {label: [{"confidence": ...}]}
                length = int(self.headers.get("Content-Length", 0))
                dets = json.loads(self.rfile.read(length) or b"{}")
                return self._json({"ok": True, "changes": fridge.on_detections(dets)})
            elif route == "sim/frame":
                length = int(self.headers.get("Content-Length", 0))
                webcam.jpeg, webcam.ts = self.rfile.read(length), time.time()
                return self._json({"ok": True})
            else:
                return self._json({"ok": False, "error": "unknown route"}, 404)
            return self._json({"ok": ok, "state": state()})

    return Handler


def fake_sensors(fridge: FridgeGuardian) -> None:
    """Fridge at about 4 °C with the door closed (no light)."""
    import math
    t = 0
    while True:
        temp = 4.0 + 0.4 * math.sin(t / 60)
        fridge.on_sensors(temp, 45.0, None)  # lux None = no light sensor in the sim
        t += 1
        time.sleep(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000,
                    help="default 8000 (macOS uses port 7000 for AirPlay Receiver)")
    ap.add_argument("--speed", type=float, default=600.0,
                    help="how much faster fridge time runs than real time (default 600: 1 s = 10 min)")
    ap.add_argument("--demo", action="store_true", help="start with a few items of different ages")
    ap.add_argument("--store", default=None, help="JSON file to keep the inventory between runs")
    args = ap.parse_args()

    clock = SimClock(args.speed)
    fridge = FridgeGuardian(clock=clock, rt_clock=time.time, store_path=args.store)
    if args.demo and not fridge.items:
        now = clock()
        for label, days_ago in [("milk", 4.2), ("broccoli", 3.1), ("cheese", 6), ("banana", 1), ("leftovers", 3.5)]:
            item = fridge.add_item(label)
            for it in fridge.items:
                if it.id == item["id"]:
                    it.added_ts = now - days_ago * 86400

    webcam = Webcam()
    camera = FakeCamera(fridge, webcam)
    threading.Thread(target=fake_sensors, args=(fridge,), daemon=True).start()

    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(fridge, camera, args.speed, webcam))
    except OSError as exc:
        sys.exit(f"Port {args.port} is busy ({exc}). Try another one: python sim/run_sim.py --port 8080")
    print(f"Fridge Guardian simulator on http://localhost:{args.port}  (time x{args.speed:g}, Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
