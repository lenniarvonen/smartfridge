"""Fridge Guardian v1 - Linux side of the Arduino UNO Q (App Lab app).

- The camera brick detects food at the fridge door, on the board.
- fridge_logic.py turns detections into an inventory with time in the fridge.
- The web page (assets/) shows each item's age against its shelf life.
- The sketch drives the LED matrix, Modulino Pixels and Buzzer.
"""

import os
import time

from arduino.app_utils import App, Bridge, Logger
from arduino.app_bricks.web_ui import WebUI
from arduino.app_bricks.video_objectdetection import VideoObjectDetection

from fridge_logic import FridgeGuardian
import icons

logger = Logger("fridge-guardian")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

fridge = FridgeGuardian(store_path=os.path.join(DATA_DIR, "inventory.json"))

# ---- Camera: food detection runs on the board ------------------------------
# The default model is a COCO object detector (banana, apple, orange, broccoli,
# carrot, sandwich, bottle, cup...). Swap in your own Edge Impulse model from the
# Brick settings in App Lab to recognise milk, cheese, leftovers and so on.
detector = VideoObjectDetection(confidence=0.5, debounce_sec=0.0)


def on_frame(detections: dict):
    changes = fridge.on_detections(detections)
    if changes:
        logger.info(f"Inventory changes: {changes}")


detector.on_detect_all(on_frame)

# ---- Sensors and buttons from the sketch -----------------------------------
Bridge.provide("sensors", lambda t, h, lux: fridge.on_sensors(t, h, lux))
Bridge.provide("button", lambda i: fridge.on_button(int(i)))

# ---- Web page + API ---------------------------------------------------------
ui = WebUI()


def _ok(result=True):
    return {"ok": bool(result), "state": fridge.snapshot()}


ui.expose_api("GET", "/api/state", lambda: fridge.snapshot())


def api_add(label: str, shelf_days: float = 0):
    return _ok(fridge.add_item(label, shelf_days or None))


def api_remove(item_id: str):
    return _ok(fridge.remove_item(item_id))


def api_shelf(item_id: str, days: float):
    return _ok(fridge.set_shelf_days(item_id, days))


def api_mode(mode: str):
    fridge.set_mode(mode)
    return _ok()


def api_clear():
    fridge.clear()
    return _ok()


def api_silence():
    fridge.silence()
    return _ok()


ui.expose_api("POST", "/api/add", api_add)
ui.expose_api("POST", "/api/remove", api_remove)
ui.expose_api("POST", "/api/shelf", api_shelf)
ui.expose_api("POST", "/api/mode", api_mode)
ui.expose_api("POST", "/api/clear", api_clear)
ui.expose_api("POST", "/api/silence", api_silence)

# ---- Outputs: LED matrix, Pixels, Buzzer ------------------------------------
_last = {"icon": None, "color": None}


def output_loop():
    out = fridge.outputs()
    try:
        if out["icon"] != _last["icon"]:
            Bridge.call("draw", icons.frame(out["icon"]))
            _last["icon"] = out["icon"]
        if out["color"] != _last["color"]:
            Bridge.call("set_pixels", *out["color"])
            _last["color"] = out["color"]
        if out["beep"]:
            Bridge.call("beep", *out["beep"])
    except Exception as exc:  # keep running if the sketch is restarting
        logger.warning(f"Output update failed: {exc}")
        _last["icon"] = _last["color"] = None
    time.sleep(0.5)


logger.info("Fridge Guardian started: open http://<board-name>.local:7000")
App.run(user_loop=output_loop)
