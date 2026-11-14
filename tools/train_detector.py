"""Fine-tune a YOLO detector on the fridge dataset and benchmark it on the test split.

Input: the output folder of tools/prepare_dataset.py (training/ + testing/ in COCO).
Output (in --work): YOLO-format data, training run, best.pt, benchmarks.json.

    pip3 install ultralytics
    python3 tools/train_detector.py ~/Downloads/fridge-ei --work ~/fridge-train --epochs 30 --hours 1

Benchmarks written to benchmarks.json:
- full test metrics of the fine-tuned model (mAP50, mAP50-95, precision, recall, per class)
- baseline: the stock COCO yolo11n vs the fine-tuned model on the classes both know
  (apple, banana, carrot), scored with the same AP50 code
- CPU inference speed of both models
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time

BASELINE_SHARED = ["apple", "banana", "carrot"]  # labels in both COCO and our dataset


# --------------------------------------------------------------------------- data
def coco_to_yolo(coco_dir: str, out_img: str, out_lbl: str, names: list[str], max_side: int,
                 subset: list | None = None) -> int:
    from PIL import Image

    d = json.load(open(os.path.join(coco_dir, "_annotations.coco.json")))
    cat_name = {c["id"]: c["name"] for c in d["categories"]}
    by_img: dict[int, list] = {}
    for a in d["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)
    images = subset if subset is not None else d["images"]
    for im in images:
        src = os.path.join(coco_dir, im["file_name"])
        stem = os.path.splitext(im["file_name"])[0]
        dst = os.path.join(out_img, stem + ".jpg")
        if not os.path.exists(dst):
            with Image.open(src) as pic:
                pic = pic.convert("RGB")
                scale = max_side / max(pic.size)
                if scale < 1:  # shrink once here, so training doesn't decode huge JPEGs every epoch
                    pic = pic.resize((round(pic.width * scale), round(pic.height * scale)))
                pic.save(dst, quality=90)
        W, H = im["width"], im["height"]
        lines = []
        for a in by_img.get(im["id"], []):
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            cls = names.index(cat_name[a["category_id"]])
            lines.append(f"{cls} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
        with open(os.path.join(out_lbl, stem + ".txt"), "w") as f:
            f.write("\n".join(lines))
    return len(images)


def build_yolo_dataset(src: str, work: str, max_side: int, val_frac: float, seed: int) -> tuple[str, list[str]]:
    train_coco = json.load(open(os.path.join(src, "training", "_annotations.coco.json")))
    names = [c["name"] for c in sorted(train_coco["categories"], key=lambda c: c["id"])]
    imgs = list(train_coco["images"])
    random.Random(seed).shuffle(imgs)
    n_val = max(1, int(len(imgs) * val_frac))
    root = os.path.join(work, "yolo")
    counts = {
        "train": coco_to_yolo(os.path.join(src, "training"), f"{root}/images/train", f"{root}/labels/train",
                              names, max_side, imgs[n_val:]),
        "val": coco_to_yolo(os.path.join(src, "training"), f"{root}/images/val", f"{root}/labels/val",
                            names, max_side, imgs[:n_val]),
        "test": coco_to_yolo(os.path.join(src, "testing"), f"{root}/images/test", f"{root}/labels/test",
                             names, max_side),
    }
    yaml = os.path.join(root, "data.yaml")
    with open(yaml, "w") as f:
        f.write(f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\n")
        f.write("names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    print(f"YOLO dataset: {counts}")
    return yaml, names


# ------------------------------------------------------------------- AP50 scoring
def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def ap50(preds: list, gts: dict) -> float:
    """preds: [(conf, img, box)], gts: {img: [box, ...]}. All-point interpolated AP at IoU 0.5."""
    n_gt = sum(len(v) for v in gts.values())
    if n_gt == 0:
        return float("nan")
    used = {k: [False] * len(v) for k, v in gts.items()}
    tp, fp = [], []
    for conf, img, box in sorted(preds, key=lambda p: -p[0]):
        best, j_best = 0.0, -1
        for j, g in enumerate(gts.get(img, [])):
            o = iou(box, g)
            if o > best:
                best, j_best = o, j
        if best >= 0.5 and not used[img][j_best]:
            used[img][j_best] = True
            tp.append(1), fp.append(0)
        else:
            tp.append(0), fp.append(1)
    ap, prev_r, ctp, cfp, prec, rec = 0.0, 0.0, 0, 0, [], []
    for t, f in zip(tp, fp):
        ctp += t
        cfp += f
        prec.append(ctp / (ctp + cfp))
        rec.append(ctp / n_gt)
    for i in range(len(prec) - 2, -1, -1):  # precision envelope
        prec[i] = max(prec[i], prec[i + 1])
    for p, r in zip(prec, rec):
        ap += p * (r - prev_r)
        prev_r = r
    return ap


def shared_class_ap(model, test_root: str, names: list[str], classes: list[str], imgsz: int) -> dict:
    """AP50 per class on test images, mapping the model's label names to ours by name."""
    img_dir, lbl_dir = f"{test_root}/images/test", f"{test_root}/labels/test"
    gts = {c: {} for c in classes}
    preds = {c: [] for c in classes}
    files = sorted(os.listdir(img_dir))
    from PIL import Image
    for fn in files:
        stem = os.path.splitext(fn)[0]
        with Image.open(os.path.join(img_dir, fn)) as pic:
            W, H = pic.size
        for line in open(os.path.join(lbl_dir, stem + ".txt")).read().split("\n"):
            if not line.strip():
                continue
            c, cx, cy, w, h = line.split()
            name = names[int(c)]
            if name in gts:
                cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
                gts[name].setdefault(stem, []).append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        r = model.predict(os.path.join(img_dir, fn), imgsz=imgsz, conf=0.001, verbose=False, device="cpu")[0]
        for cls, conf, box in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
            name = r.names[int(cls)]
            if name in preds:
                preds[name].append((conf, stem, tuple(box)))
    return {c: ap50(preds[c], gts[c]) for c in classes}


def cpu_speed_ms(model, img_dir: str, imgsz: int, n: int = 60) -> float:
    files = sorted(os.listdir(img_dir))[:n]
    model.predict(os.path.join(img_dir, files[0]), imgsz=imgsz, verbose=False, device="cpu")  # warm-up
    t0 = time.perf_counter()
    for fn in files:
        model.predict(os.path.join(img_dir, fn), imgsz=imgsz, verbose=False, device="cpu")
    return (time.perf_counter() - t0) / len(files) * 1000


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="output of tools/prepare_dataset.py")
    ap.add_argument("--work", default="fridge-train")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hours", type=float, default=None, help="time budget; stops training early")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu, mps (Apple GPU) or 0 (CUDA)")
    ap.add_argument("--skip-train", action="store_true", help="only evaluate an existing run")
    args = ap.parse_args()

    from ultralytics import YOLO

    work = os.path.abspath(os.path.expanduser(args.work))
    os.makedirs(work, exist_ok=True)
    yaml, names = build_yolo_dataset(os.path.expanduser(args.src), work, max(args.imgsz * 2, 416),
                                     args.val_frac, args.seed)
    run_dir = os.path.join(work, "run")
    if not args.skip_train:
        model = YOLO(args.model)
        kw = dict(data=yaml, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, workers=args.workers,
                  device=args.device, project=work, name="run", exist_ok=True, seed=args.seed,
                  plots=True, cache=False, patience=10, verbose=False)
        if args.hours:
            kw["time"] = args.hours
        t0 = time.time()
        model.train(**kw)
        train_minutes = (time.time() - t0) / 60
    else:
        train_minutes = None

    best = os.path.join(run_dir, "weights", "best.pt")
    tuned = YOLO(best)
    m = tuned.val(data=yaml, split="test", imgsz=args.imgsz, batch=args.batch, device=args.device,
                  project=work, name="test", exist_ok=True, plots=True, verbose=False)
    per_class = {names[c]: {"ap50": float(m.box.ap50[i]), "ap50_95": float(m.box.ap[i]),
                            "precision": float(m.box.p[i]), "recall": float(m.box.r[i])}
                 for i, c in enumerate(m.box.ap_class_index)}

    yolo_root = os.path.join(work, "yolo")
    test_img_dir = os.path.join(yolo_root, "images", "test")
    test_boxes = {}
    for fn in os.listdir(os.path.join(yolo_root, "labels", "test")):
        for line in open(os.path.join(yolo_root, "labels", "test", fn)).read().split("\n"):
            if line.strip():
                n = names[int(line.split()[0])]
                test_boxes[n] = test_boxes.get(n, 0) + 1

    print("Scoring the stock COCO model vs fine-tuned on shared classes...")
    stock = YOLO(args.model)
    baseline = {
        "classes": BASELINE_SHARED,
        "stock_coco": shared_class_ap(stock, yolo_root, names, BASELINE_SHARED, args.imgsz),
        "fine_tuned": shared_class_ap(tuned, yolo_root, names, BASELINE_SHARED, args.imgsz),
    }
    speed = {"stock_coco_ms": cpu_speed_ms(stock, test_img_dir, args.imgsz),
             "fine_tuned_ms": cpu_speed_ms(tuned, test_img_dir, args.imgsz)}

    epochs_done = None
    results_csv = os.path.join(run_dir, "results.csv")
    if os.path.exists(results_csv):
        epochs_done = max(0, len(open(results_csv).read().strip().split("\n")) - 1)
    out = {
        "model": args.model, "imgsz": args.imgsz, "device": args.device,
        "epochs_requested": args.epochs, "epochs_done": epochs_done, "train_minutes": train_minutes,
        "test": {"images": len(os.listdir(test_img_dir)), "boxes": sum(test_boxes.values()),
                 "map50": float(m.box.map50), "map50_95": float(m.box.map),
                 "precision": float(m.box.mp), "recall": float(m.box.mr)},
        "per_class": {k: {**v, "test_boxes": test_boxes.get(k, 0)} for k, v in per_class.items()},
        "baseline": baseline, "speed": speed,
        "best_weights": best, "results_csv": results_csv,
    }
    with open(os.path.join(work, "benchmarks.json"), "w") as f:
        json.dump(out, f, indent=2)
    shutil.copy2(best, os.path.join(work, "fridge_yolo11n.pt"))
    print(json.dumps(out["test"], indent=2))
    print(f"Saved {work}/benchmarks.json and {work}/fridge_yolo11n.pt")


if __name__ == "__main__":
    main()
