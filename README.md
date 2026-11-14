# Fridge Guardian v1

Edge AI fridge add-on for the Arduino UNO Q. A camera at the fridge door recognises food going in and out, and the web page shows how long each item has been in the fridge compared with its shelf life. "Eat first" goes to the item with the least time left. All detection runs on the board, and no images leave the fridge.

Built for Junction 2026, "Make Europe Green through Edge AI using Arduino".

## What's in this folder

| Path | What it is |
| --- | --- |
| `app/` | The Arduino App Lab app. Copy this to the UNO Q. |
| `app/sketch/sketch.ino` | Microcontroller side: Modulino sensors, buttons, Pixels, buzzer, LED matrix |
| `app/python/main.py` | Linux side: camera brick, web API, outputs |
| `app/python/fridge_logic.py` | Inventory and shelf-life logic (plain Python, also used by the simulator and tests) |
| `app/assets/` | Web page served on port 7000 |
| `sim/run_sim.py` | Runs the same logic and web page on a laptop, no board needed |
| `tests/` | Unit tests for the logic |

## 1. Try it on your laptop now (no hardware)

Needs Python 3.10 or newer, nothing else.

```bash
python sim/run_sim.py --demo
```

Open http://localhost:7000. Click the simulator buttons to "show" items to the camera. Fridge time runs 600x faster than real time (change with `--speed`), so you can watch items move from fresh to eat soon to past shelf life within a few minutes.

Run the tests with `python -m pytest tests`.

## 2. Run it on the UNO Q

1. In Arduino App Lab, create a new app called Fridge Guardian.
2. Copy `app/sketch/sketch.ino`, `app/sketch/sketch.yaml`, everything in `app/python/` and everything in `app/assets/` into the matching folders of the new app. Replace `app.yaml` with ours. It adds the **WebUI - HTML** and **Video Object Detection** bricks.
3. Install the **Arduino_Modulino** library if App Lab asks for it.
4. Plug a USB webcam into the UNO Q through a powered USB-C hub, and connect the Modulinos with Qwiic cables.
5. Click **Run**, then open `http://<board-name>.local:7000` from a laptop on the same network. If the page doesn't load over Wi-Fi, connect App Lab to the board over USB.

### Hardware (all optional except the camera)

| Part | Role in v1 |
| --- | --- |
| USB webcam | Recognises items at the door (required) |
| Modulino Light | Door open detection: the camera only adds items while the door is open, and an alarm sounds after 60 s open |
| Modulino Thermo | Fridge temperature and humidity on the page |
| Modulino Buttons | A = putting in, B = taking out (next 15 s), C = silence alarm |
| Modulino Pixels | Status light: green fresh, orange eat soon, red past shelf life, blue door open |
| Modulino Buzzer | Door-open alarm |
| Built-in LED matrix | Status icon |

Any missing Modulino is skipped, and the sketch keeps retrying, so you can plug them in while it runs.

## How it works

- **Detection:** the default model on the board is a COCO object detector. It already knows banana, apple, orange, broccoli, carrot, sandwich, pizza, cake, bottle and cup. A label must be seen in 3 frames within 2 seconds before it counts. Then it is ignored for 6 seconds, so one item isn't added twice.
- **In or out:** by default a detected item is added. Press "Taking out" on the page, or button B, and the next detection removes the oldest item with that label.
- **Shelf life:** each label has a default number of days in `SHELF_LIFE_DAYS` in `fridge_logic.py`. Click the number on any item to change it. An item is "eat soon" when under 25% of its shelf life, or under 1 day, is left.
- **Persistence:** the inventory is saved to `app/data/inventory.json` on the board, so it survives restarts.

The shelf life values are rough defaults for ranking what to eat first, not food-safety advice.

## Next step: your own food model

To recognise milk cartons, cheese, yogurt or leftover boxes, train an object detection model in Edge Impulse:

1. Start from public fridge datasets, for example [aicook / FridgeFoodDetection](https://universe.roboflow.com/objectdetectionlearning-4empo/fridgefooddetection-cxnl3) and [FridgeItems_2](https://universe.roboflow.com/capstoneproject-yly17/fridgeitems_2) on Roboflow. Check each licence.
2. Add 30 to 50 photos per item from your own fridge door camera.
3. Train an object detection model in Edge Impulse and deploy it to the UNO Q from App Lab. Then pick it in the Video Object Detection brick's settings.
4. Make sure the label names match the keys in `SHELF_LIFE_DAYS`, or add new keys.

No code changes are needed: the logic works with whatever labels the model gives.

## Roadmap after v1

- Spoilage gas sensor (BME688 or MQ-series) to catch food that goes off early
- Compressor vibration anomaly detection with Modulino Movement and the `vibration_anomaly_detection` brick
