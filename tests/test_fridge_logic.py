import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app", "python"))

from fridge_logic import FridgeGuardian  # noqa: E402

DAY = 86400.0


class Clock:
    def __init__(self):
        self.t = 1_700_000_000.0

    def __call__(self):
        return self.t

    def tick(self, s):
        self.t += s


def frames(fridge, clock, label, n=3, dt=0.2):
    for _ in range(n):
        fridge.on_detections({label: [{"confidence": 0.9}]})
        clock.tick(dt)


def test_item_added_after_consistent_frames_only():
    c = Clock()
    f = FridgeGuardian(clock=c)
    f.on_detections({"banana": [{"confidence": 0.9}]})
    assert f.items == []                      # one frame is not enough
    frames(f, c, "banana", n=2)
    assert [i.label for i in f.items] == ["banana"]
    frames(f, c, "banana", n=3)               # cooldown: no duplicate
    assert len(f.items) == 1


def test_non_food_ignored():
    c = Clock()
    f = FridgeGuardian(clock=c)
    frames(f, c, "person", n=5)
    assert f.items == []


def test_states_follow_time_in_fridge():
    c = Clock()
    f = FridgeGuardian(clock=c)
    f.add_item("milk")                       # 5 days shelf life
    assert f.snapshot()["items"][0]["state"] == "fresh"
    c.tick(4.2 * DAY)
    item = f.snapshot()["items"][0]
    assert item["state"] == "eat_soon"
    assert abs(item["age_days"] - 4.2) < 1e-3
    c.tick(1 * DAY)
    assert f.snapshot()["items"][0]["state"] == "expired"
    assert f.summary()["level"] == "expired"


def test_eat_first_is_least_time_left():
    c = Clock()
    f = FridgeGuardian(clock=c)
    f.add_item("cheese")                     # 21 days
    c.tick(DAY)
    f.add_item("sandwich")                   # 2 days
    assert f.summary()["eat_first"]["label"] == "sandwich"


def test_take_out_mode_removes_oldest():
    c = Clock()
    f = FridgeGuardian(clock=c)
    f.add_item("apple")
    c.tick(10)
    f.add_item("apple")
    oldest = min(f.items, key=lambda i: i.added_ts).id
    f.set_mode("out")
    frames(f, c, "apple")
    assert len(f.items) == 1 and f.items[0].id != oldest
    c.tick(30)
    assert f.snapshot()["mode"] == "in"     # take-out mode times out


def test_door_gates_detection_and_alarm():
    c = Clock()
    f = FridgeGuardian(clock=c)
    f.on_sensors(4.0, 50.0, 2.0)            # dark: door closed
    frames(f, c, "carrot")
    assert f.items == []
    f.on_sensors(4.0, 50.0, 200.0)          # light: door open
    frames(f, c, "carrot")
    assert len(f.items) == 1
    c.tick(61)
    f.on_sensors(4.0, 50.0, 200.0)
    assert f.summary()["level"] == "door"
    assert f.outputs()["beep"] is not None
    assert f.outputs()["beep"] is None       # rate limited


def test_persistence_roundtrip():
    c = Clock()
    path = os.path.join(tempfile.mkdtemp(), "inv.json")
    f = FridgeGuardian(clock=c, store_path=path)
    f.add_item("yogurt", shelf_days=8)
    g = FridgeGuardian(clock=c, store_path=path)
    assert [(i.label, i.shelf_days) for i in g.items] == [("yogurt", 8.0)]


def test_icons_are_full_frames():
    import icons
    for name in ["fresh", "eat_soon", "expired", "door", "empty"]:
        fr = icons.frame(name)
        assert len(fr) == 104 and max(fr) == 7
