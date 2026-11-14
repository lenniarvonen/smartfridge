"""Real object detection with your laptop webcam, feeding the simulator.

Runs a YOLO object detector on the webcam and sends every frame's detections
to sim/run_sim.py, exactly like the camera brick does on the UNO Q.

The default model (yolo11n.pt, downloaded automatically, about 6 MB) knows the
80 COCO objects. Food it can see: banana, apple, orange, broccoli, carrot,
sandwich, hot dog, pizza, donut, cake, plus bottle, cup and bowl. That's close
to the default detector on the UNO Q.

Setup (once):
    pip3 install ultralytics opencv-python

Run (two terminals):
    python3 sim/run_sim.py
    python3 sim/webcam_detector.py

On macOS, allow camera access for your Terminal app when asked
(System Settings > Privacy & Security > Camera).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

try:
    import cv2
    from ultralytics import YOLO
except ImportError:
    sys.exit("Missing packages. Install them with:\n    pip3 install ultralytics opencv-python")

FOOD_AND_CONTAINERS = {
    "banana", "apple", "orange", "broccoli", "carrot", "sandwich", "hot dog",
    "pizza", "donut", "cake", "bottle", "cup", "bowl", "wine glass",
}


def post(url: str, data: bytes, content_type: str) -> dict | None:
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(req, timeout=2) as res:
            return json.loads(res.read() or b"{}")
    except OSError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", default="http://localhost:8000", help="simulator address")
    ap.add_argument("--model", default="yolo11n.pt",
                    help="YOLO weights: default COCO model, or your own trained .pt")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (try 1 for an external camera)")
    ap.add_argument("--conf", type=float, default=0.5, help="confidence threshold (UNO Q app uses 0.5)")
    ap.add_argument("--all-labels", action="store_true",
                    help="send every label, not just food (the app ignores person, chair etc. anyway)")
    ap.add_argument("--window", action="store_true", help="also show a local preview window")
    args = ap.parse_args()

    model = YOLO(args.model)
    custom = args.model != "yolo11n.pt"
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("Could not open the webcam. Check camera permission for your terminal, or try --camera 1.")

    print(f"Detecting with {args.model}. Hold food in front of the camera. Ctrl+C to stop.")
    print(f"Watch the page at {args.sim}")
    last_print, last_frame_post, offline_warned = 0.0, 0.0, False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            result = model(frame, conf=args.conf, verbose=False)[0]
            detections: dict[str, list] = {}
            for cls, conf, box in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(),
                                      result.boxes.xyxy.tolist()):
                label = result.names[int(cls)]
                if custom or args.all_labels or label in FOOD_AND_CONTAINERS:
                    detections.setdefault(label, []).append(
                        {"confidence": round(conf, 3), "bounding_box_xyxy": [round(v) for v in box]})

            res = post(f"{args.sim}/api/sim/detections", json.dumps(detections).encode(), "application/json")
            if res is None and not offline_warned:
                print(f"Simulator not reachable at {args.sim}. Is sim/run_sim.py running?")
                offline_warned = True
            elif res is not None:
                offline_warned = False
                for change in res.get("changes", []):
                    print(("Added " if change.startswith("+") else "Took out ") + change[1:])

            annotated = result.plot()
            now = time.time()
            if now - last_frame_post > 0.1:
                small = cv2.resize(annotated, (640, int(640 * annotated.shape[0] / annotated.shape[1])))
                ok_jpg, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok_jpg:
                    post(f"{args.sim}/api/sim/frame", jpg.tobytes(), "image/jpeg")
                last_frame_post = now
            if detections and now - last_print > 2:
                print("Sees: " + ", ".join(f"{k} {max(d['confidence'] for d in v):.0%}" for k, v in detections.items()))
                last_print = now
            if args.window:
                cv2.imshow("Fridge Guardian webcam", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
