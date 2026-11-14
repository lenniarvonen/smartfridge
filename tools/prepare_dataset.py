"""Turn the Roboflow FridgeFoodDetection COCO export into a smaller,
fridge-only dataset for Edge Impulse.

- keeps only food that lives in a fridge, and merges duplicate labels
  (heavy_cream + fresh cream -> cream, beef + ground_beef + meat -> meat, ...)
- removes boxes for pantry items (flour, sugar, potato...); the images stay,
  so the model learns to ignore those items
- writes training/ (train + valid) and testing/ (test), each with
  _annotations.coco.json, the layout the Edge Impulse uploader expects
- optional --max-images to keep Edge Impulse training within free-tier time

Standard library only:

    python tools/prepare_dataset.py ~/Downloads/FridgeFoodDetection ~/Downloads/fridge-ei
    python tools/prepare_dataset.py SRC OUT --max-images 1500
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys

# Source label -> label used by the app (None = drop)
LABEL_MAP = {
    "milk": "milk",
    "cheese": "cheese", "goat_cheese": "cheese",
    "butter": "butter",
    "egg": "egg",
    "heavy_cream": "cream", "fresh cream": "cream",
    "chicken": "chicken",
    "beef": "meat", "ground_beef": "meat", "meat": "meat",
    "ham": "ham",
    "shrimp": "shrimp",
    "tomato": "tomato",
    "strawberry": "strawberry",
    "blueberry": "blueberry",
    "carrot": "carrot",
    "green_beans": "green_beans",
    "mushroom": "mushroom",
    "spinach": "leafy_greens", "green leaves": "leafy_greens",
    "cabbage": "cabbage",
    "cucumber": "cucumber",
    "capsicum": "bell_pepper",
    "eggplant": "eggplant",
    "lime": "citrus", "lemon": "citrus",
    "apple": "apple",
    "banana": "banana",
    # Pantry or rarely in the fridge: dropped
    "flour": None, "sugar": None, "chocolate": None, "bread": None,
    "potato": None, "sweet_potato": None, "onion": None, "ginger": None,
    "corn": None, "green chilly": None,
}

SPLITS = {"training": ["train", "valid"], "testing": ["test"]}


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def convert(src: str, out: str, max_images: int | None, keep_empty: float, seed: int,
            max_test: int | None = None) -> dict:
    rnd = random.Random(seed)
    new_names = sorted({v for v in LABEL_MAP.values() if v})
    new_ids = {name: i for i, name in enumerate(new_names)}
    categories = [{"id": i, "name": n, "supercategory": "food"} for n, i in new_ids.items()]
    report = {}

    for out_split, in_splits in SPLITS.items():
        images, annotations = [], []
        for in_split in in_splits:
            ann_path = os.path.join(src, in_split, "_annotations.coco.json")
            if not os.path.exists(ann_path):
                continue
            d = load(ann_path)
            names = {c["id"]: c["name"] for c in d["categories"]}
            used = {a["category_id"] for a in d["annotations"]}
            unknown = {names[c] for c in used if names.get(c) not in LABEL_MAP}
            if unknown:  # Roboflow's unused parent class is skipped silently
                print(f"note: unmapped labels in {in_split}, dropped: {sorted(unknown)}", file=sys.stderr)
            by_image: dict[int, list] = {}
            for a in d["annotations"]:
                target = LABEL_MAP.get(names.get(a["category_id"], ""))
                if target:
                    by_image.setdefault(a["image_id"], []).append((a, target))
            for im in d["images"]:
                boxes = by_image.get(im["id"], [])
                if not boxes and rnd.random() > keep_empty:
                    continue
                images.append((in_split, im, boxes))

        limit = max_images if out_split == "training" else max_test
        if limit and len(images) > limit:
            rnd.shuffle(images)
            images = images[:limit]

        split_dir = os.path.join(out, out_split)
        os.makedirs(split_dir, exist_ok=True)
        coco_images, counts = [], {}
        for new_id, (in_split, im, boxes) in enumerate(images):
            fname = f"{in_split}_{im['file_name']}"
            src_img = os.path.join(src, in_split, im["file_name"])
            if os.path.exists(src_img):
                shutil.copy2(src_img, os.path.join(split_dir, fname))
            coco_images.append({"id": new_id, "file_name": fname,
                                "width": im["width"], "height": im["height"]})
            for a, target in boxes:
                annotations.append({"id": len(annotations), "image_id": new_id,
                                    "category_id": new_ids[target], "bbox": a["bbox"]})
                counts[target] = counts.get(target, 0) + 1

        with open(os.path.join(split_dir, "_annotations.coco.json"), "w") as f:
            json.dump({"info": {"description": "FridgeFoodDetection, fridge-only subset"},
                       "categories": categories, "images": coco_images,
                       "annotations": annotations}, f)
        report[out_split] = {"images": len(coco_images), "boxes": len(annotations),
                             "per_label": dict(sorted(counts.items(), key=lambda x: -x[1]))}
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="unzipped Roboflow COCO export (has train/ valid/ test/)")
    ap.add_argument("out", help="output folder to create")
    ap.add_argument("--max-images", type=int, default=None,
                    help="cap training images; try 1500 on the Edge Impulse free plan")
    ap.add_argument("--max-test", type=int, default=None, help="cap testing images (default: all)")
    ap.add_argument("--keep-empty", type=float, default=0.1,
                    help="share of images with no fridge food to keep as background (default 0.1)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, out = os.path.expanduser(args.src), os.path.expanduser(args.out)
    if not os.path.isdir(os.path.join(src, "train")):
        sys.exit(f"{src} has no train/ folder. Point this at the unzipped Roboflow COCO export.")
    report = convert(src, out, args.max_images, args.keep_empty, args.seed, args.max_test)
    for split, r in report.items():
        print(f"{split}: {r['images']} images, {r['boxes']} boxes")
        print("   " + ", ".join(f"{k} {v}" for k, v in r["per_label"].items()))
    print(f"\nDone. Upload the folder {out} in Edge Impulse: Data acquisition > Upload data.")


if __name__ == "__main__":
    main()
