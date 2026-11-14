"""Fridge Guardian v1 core logic: item tracking + time in fridge vs shelf life.

Pure Python with no Arduino imports, so the same code runs on the UNO Q
(main.py), on a laptop (sim/run_sim.py) and in the tests.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass

# Rough fridge shelf life in days, after the item goes into the fridge.
# Conservative rule-of-thumb defaults to rank "eat me first" - not food-safety
# advice. Edit freely, or override per item from the web page.
SHELF_LIFE_DAYS = {
    # Labels from the custom Edge Impulse model (tools/prepare_dataset.py)
    "milk": 5, "cheese": 14, "butter": 30, "egg": 21, "cream": 5,
    "chicken": 2, "meat": 2, "ham": 4, "shrimp": 2,
    "tomato": 7, "strawberry": 3, "blueberry": 7, "apple": 21, "banana": 5,
    "citrus": 21, "carrot": 21, "green_beans": 5, "mushroom": 5,
    "leafy_greens": 4, "cabbage": 14, "cucumber": 7, "bell_pepper": 7, "eggplant": 5,
    # Labels the default COCO detector on the UNO Q can see
    "orange": 14, "broccoli": 4, "sandwich": 2, "hot dog": 3, "pizza": 3,
    "donut": 3, "cake": 4, "bottle": 7, "cup": 2, "bowl": 3,
    # Other common items you can add by hand
    "yogurt": 10, "eggs": 21, "salad": 4, "leftovers": 3, "fish": 2, "juice": 7,
}
DEFAULT_SHELF_DAYS = 5.0

# Detector labels that are never food
IGNORED_LABELS = {"person", "refrigerator", "dining table", "chair", "cell phone",
                  "hand", "laptop", "tv", "background", "unknown", "empty", "oven",
                  "microwave", "sink", "book", "remote", "keyboard", "mouse"}

STATES = ["fresh", "eat_soon", "expired"]


@dataclass
class Config:
    detect_min_hits: int = 3          # a label must be seen in this many frames ...
    detect_window_s: float = 2.0      # ... within this many seconds to count
    detect_cooldown_s: float = 6.0    # then ignore that label for this long
    eat_soon_fraction: float = 0.25   # under 25% of shelf life left = eat soon
    eat_soon_min_days: float = 1.0    # ... or under 1 day left
    door_lux_threshold: float = 15.0  # interior light above this = door open
    door_alarm_s: float = 60.0        # door open longer than this = alarm
    take_out_mode_s: float = 15.0     # "take out" mode lasts this long
    beep_every_s: float = 10.0        # minimum gap between alarm beeps
    only_detect_when_door_open: bool = True  # if a light sensor is present


@dataclass
class Item:
    id: str
    label: str
    added_ts: float
    shelf_days: float

    def age_days(self, now: float) -> float:
        return max(0.0, (now - self.added_ts) / 86400.0)

    def days_left(self, now: float) -> float:
        return self.shelf_days - self.age_days(now)

    def state(self, now: float, cfg: Config) -> str:
        left = self.days_left(now)
        if left <= 0:
            return "expired"
        if left < max(cfg.eat_soon_min_days, cfg.eat_soon_fraction * self.shelf_days):
            return "eat_soon"
        return "fresh"


class FridgeGuardian:
    def __init__(self, config: Config | None = None, clock=time.time,
                 store_path: str | None = None, rt_clock=None):
        self.cfg = config or Config()
        self.clock = clock              # food time: how long items have been in the fridge
        self.rt = rt_clock or clock     # real time: debounce, door alarm, modes (differs only in the simulator)
        self.store_path = store_path
        self.lock = threading.RLock()
        self.items: list[Item] = []
        self.events: deque[dict] = deque(maxlen=30)
        self.temp_c: float | None = None
        self.humidity: float | None = None
        self.lux: float | None = None
        self.sensors_ts = 0.0
        self.door_open = False
        self.door_open_since: float | None = None
        self.mode = "in"
        self.mode_until = 0.0
        self.silenced_until = 0.0
        self.last_seen: dict[str, float] = {}   # label -> confidence of last frame
        self._hits: dict[str, deque[float]] = {}
        self._cooldown: dict[str, float] = {}
        self._last_beep = 0.0
        self._load()

    # ------------------------------------------------------------------ inputs
    def on_detections(self, detections: dict) -> list[str]:
        """Feed one frame of detections: {label: [{"confidence": float, ...}, ...]}.

        Returns the list of inventory changes made (for logging)."""
        now = self.rt()
        food_now = self.clock()
        changes = []
        with self.lock:
            self.last_seen = {
                str(k).lower(): max((d.get("confidence", 0) for d in v), default=0)
                if isinstance(v, list) else 0
                for k, v in detections.items()
            }
            if (self.cfg.only_detect_when_door_open and self.lux is not None
                    and not self.door_open):
                return changes
            for raw in detections:
                label = str(raw).strip().lower()
                if label in IGNORED_LABELS or now < self._cooldown.get(label, 0.0):
                    continue
                hits = self._hits.setdefault(label, deque())
                hits.append(now)
                while hits and now - hits[0] > self.cfg.detect_window_s:
                    hits.popleft()
                if len(hits) >= self.cfg.detect_min_hits:
                    hits.clear()
                    self._cooldown[label] = now + self.cfg.detect_cooldown_s
                    if self._current_mode(now) == "out":
                        if self._remove_label(label, food_now):
                            changes.append(f"-{label}")
                    else:
                        self._add(label, food_now)
                        changes.append(f"+{label}")
            if changes:
                self._save()
        return changes

    def on_sensors(self, temp_c: float | None, humidity: float | None, lux: float | None) -> None:
        now = self.rt()
        with self.lock:
            self.temp_c = None if temp_c is None or temp_c <= -900 else round(float(temp_c), 1)
            self.humidity = None if humidity is None or humidity <= -900 else round(float(humidity), 1)
            self.lux = None if lux is None or lux < 0 else float(lux)
            self.sensors_ts = now
            if self.lux is not None:
                is_open = self.lux > self.cfg.door_lux_threshold
                if is_open and not self.door_open:
                    self.door_open_since = now
                if not is_open:
                    self.door_open_since = None
                self.door_open = is_open

    def on_button(self, index: int) -> None:
        """Modulino Buttons: A = put in, B = take out, C = silence alarm."""
        if index == 0:
            self.set_mode("in")
        elif index == 1:
            self.set_mode("out")
        elif index == 2:
            self.silence()

    # ------------------------------------------------------------ UI actions
    def set_mode(self, mode: str) -> None:
        with self.lock:
            self.mode = "out" if mode == "out" else "in"
            self.mode_until = self.rt() + self.cfg.take_out_mode_s

    def add_item(self, label: str, shelf_days: float | None = None) -> dict | None:
        with self.lock:
            item = self._add(label.strip().lower(), self.clock(), shelf_days)
            self._save()
            return asdict(item) if item else None

    def remove_item(self, item_id: str) -> bool:
        with self.lock:
            for it in self.items:
                if it.id == item_id:
                    self.items.remove(it)
                    self._event(f"Removed {it.label}")
                    self._save()
                    return True
            return False

    def set_shelf_days(self, item_id: str, days: float) -> bool:
        with self.lock:
            for it in self.items:
                if it.id == item_id:
                    it.shelf_days = max(0.1, float(days))
                    self._save()
                    return True
            return False

    def clear(self) -> None:
        with self.lock:
            self.items.clear()
            self._event("Inventory cleared")
            self._save()

    def silence(self, seconds: float = 300) -> None:
        with self.lock:
            self.silenced_until = self.rt() + seconds

    # ---------------------------------------------------------------- outputs
    def summary(self) -> dict:
        now = self.clock()
        rt_now = self.rt()
        with self.lock:
            counts = {s: 0 for s in STATES}
            for it in self.items:
                counts[it.state(now, self.cfg)] += 1
            first = min(self.items, key=lambda i: i.days_left(now), default=None)
            door_alarm = bool(self.door_open_since and rt_now - self.door_open_since > self.cfg.door_alarm_s)
            if door_alarm:
                level = "door"
            elif counts["expired"]:
                level = "expired"
            elif counts["eat_soon"]:
                level = "eat_soon"
            elif self.items:
                level = "fresh"
            else:
                level = "empty"
            return {"level": level, "counts": counts, "door_alarm": door_alarm,
                    "eat_first": self._item_dict(first, now)}

    def outputs(self) -> dict:
        """What the LED matrix, Pixels and buzzer should show right now."""
        now = self.rt()
        level = self.summary()["level"]
        color = {"empty": (0, 0, 40), "fresh": (0, 180, 0), "eat_soon": (255, 140, 0),
                 "expired": (255, 0, 0), "door": (0, 80, 255)}[level]
        beep = None
        with self.lock:
            if level == "door" and now >= self.silenced_until and now - self._last_beep >= self.cfg.beep_every_s:
                self._last_beep = now
                beep = (2000, 300)
        return {"icon": level, "color": color, "beep": beep}

    def snapshot(self) -> dict:
        now = self.clock()
        rt_now = self.rt()
        with self.lock:
            items = sorted(self.items, key=lambda i: i.days_left(now))
            return {
                "now": now,
                "summary": self.summary(),
                "mode": self._current_mode(rt_now),
                "mode_s_left": max(0, self.mode_until - rt_now) if self.mode == "out" else 0,
                "door_open": self.door_open,
                "door_open_s": round(rt_now - self.door_open_since) if self.door_open_since else 0,
                "sensors": {"temp_c": self.temp_c, "humidity": self.humidity, "lux": self.lux,
                            "age_s": round(rt_now - self.sensors_ts, 1) if self.sensors_ts else None},
                "camera": self.last_seen,
                "items": [self._item_dict(i, now) for i in items],
                "events": list(reversed(self.events)),
                "shelf_life_table": SHELF_LIFE_DAYS,
            }

    # --------------------------------------------------------------- internal
    def _current_mode(self, now: float) -> str:
        if self.mode == "out" and now > self.mode_until:
            self.mode = "in"
        return self.mode

    def _add(self, label: str, now: float, shelf_days: float | None = None) -> Item | None:
        if not label or label in IGNORED_LABELS:
            return None
        days = float(shelf_days) if shelf_days else float(SHELF_LIFE_DAYS.get(label, DEFAULT_SHELF_DAYS))
        item = Item(uuid.uuid4().hex[:8], label, now, days)
        self.items.append(item)
        self._event(f"Added {label} ({days:g} days shelf life)")
        return item

    def _remove_label(self, label: str, now: float) -> bool:
        # Taking out: remove the oldest item with that label
        matches = sorted((i for i in self.items if i.label == label), key=lambda i: i.added_ts)
        if not matches:
            return False
        self.items.remove(matches[0])
        self._event(f"Took out {label}")
        return True

    def _item_dict(self, item: Item | None, now: float) -> dict | None:
        if item is None:
            return None
        return {"id": item.id, "label": item.label, "added_ts": item.added_ts,
                "age_days": round(item.age_days(now), 3), "shelf_days": item.shelf_days,
                "days_left": round(item.days_left(now), 3),
                "used_fraction": round(min(1.0, item.age_days(now) / item.shelf_days), 3),
                "state": item.state(now, self.cfg)}

    def _event(self, text: str) -> None:
        self.events.append({"ts": self.clock(), "text": text})

    # Persistence: the inventory survives app restarts
    def _save(self) -> None:
        if not self.store_path:
            return
        tmp = self.store_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump([asdict(i) for i in self.items], f)
        os.replace(tmp, self.store_path)

    def _load(self) -> None:
        if not self.store_path or not os.path.exists(self.store_path):
            return
        try:
            with open(self.store_path) as f:
                self.items = [Item(**d) for d in json.load(f)]
        except (OSError, ValueError, TypeError):
            self.items = []
