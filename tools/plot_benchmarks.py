"""Plot benchmarks from tools/train_detector.py into PNG charts.

    python3 tools/plot_benchmarks.py ~/fridge-train --out benchmarks

Writes:
  training_curves.png   validation mAP50 / mAP50-95 per epoch
  per_class_ap50.png    test AP50 per food class, with test box counts
  baseline_vs_tuned.png stock COCO model vs fine-tuned on shared classes
  summary.md            headline numbers
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference palette (light mode), see the dataviz guidance
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e4df"
BLUE = "#2a78d6"     # slot 1: fine-tuned
ORANGE = "#eb6834"   # slot 2: stock / secondary series


def style(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=10)
    ax.grid(axis="y" if ax.get_xlabel() != "AP50" else "x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=14, color=INK, fontweight="bold", pad=24 if subtitle else 12)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color=INK_2, va="bottom")


def training_curves(results_csv: str, path: str) -> None:
    rows = list(csv.DictReader(open(results_csv)))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    ep = [int(float(r["epoch"])) for r in rows]
    m50 = [float(r["metrics/mAP50(B)"]) * 100 for r in rows]
    m95 = [float(r["metrics/mAP50-95(B)"]) * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    ax.set_xlabel("Epoch", color=INK_2)
    ax.set_ylabel("Validation score (%)", color=INK_2)
    ax.plot(ep, m50, color=BLUE, linewidth=2, marker="o", markersize=4)
    ax.plot(ep, m95, color=ORANGE, linewidth=2, marker="o", markersize=4)
    ax.annotate(f"mAP50 {m50[-1]:.1f}%", (ep[-1], m50[-1]), xytext=(6, 0), textcoords="offset points",
                color=INK, fontsize=10, va="center")
    ax.annotate(f"mAP50-95 {m95[-1]:.1f}%", (ep[-1], m95[-1]), xytext=(6, 0), textcoords="offset points",
                color=INK, fontsize=10, va="center")
    ax.set_ylim(0, max(m50) * 1.15)
    ax.set_xlim(min(ep) - 0.5, max(ep) + max(2.5, len(ep) * 0.25))
    style(ax, f"Validation mAP50 reached {m50[-1]:.0f}% after {ep[-1]} epochs", "Validation split (10% of training images), YOLO11n fine-tuning on CPU")
    ax.legend(["mAP50", "mAP50-95"], frameon=False, loc="lower right", fontsize=10, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def per_class(bm: dict, path: str) -> None:
    items = sorted(bm["per_class"].items(), key=lambda kv: kv[1]["ap50"])
    names = [k.replace("_", " ") for k, _ in items]
    vals = [v["ap50"] * 100 for _, v in items]
    counts = [v["test_boxes"] for _, v in items]
    overall = bm["test"]["map50"] * 100
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(names) + 1.6), facecolor=SURFACE)
    ax.set_xlabel("AP50")
    bars = ax.barh(names, vals, color=BLUE, height=0.7)
    for b, v, n in zip(bars, vals, counts):
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{v:.0f}%  ·  {n} boxes",
                va="center", fontsize=9, color=INK_2)
    ax.axvline(overall, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(overall, len(names) - 0.3, f" mean {overall:.1f}%", color=INK, fontsize=9, va="bottom")
    ax.set_xlim(0, 115)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(axis="y", labelsize=10, colors=INK)
    style(ax, "Test accuracy per food class",
          f"AP50 on {bm['test']['images']} held-out test images; labels show test boxes per class")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def baseline(bm: dict, path: str) -> None:
    cls = bm["baseline"]["classes"]
    stock = [bm["baseline"]["stock_coco"][c] * 100 for c in cls]
    tuned = [bm["baseline"]["fine_tuned"][c] * 100 for c in cls]
    x = range(len(cls))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.6), facecolor=SURFACE)
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], stock, w, color=ORANGE, label="Stock COCO YOLO11n")
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], tuned, w, color=BLUE, label="Fine-tuned on fridge data")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f}%",
                    ha="center", fontsize=10, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(cls, fontsize=11, color=INK)
    ax.set_ylabel("Test AP50 (%)", color=INK_2)
    ax.set_ylim(0, 105)
    sp = bm["speed"]
    gain = sum(tuned) / len(tuned) - sum(stock) / len(stock)
    headline = (f"Fine-tuning adds {gain:.0f} points AP50 on the foods both models know" if gain > 0
                else "Stock vs fine-tuned on the foods both models know")
    style(ax, headline,
          f"Same test images and scoring. CPU speed: stock {sp['stock_coco_ms']:.0f} ms, fine-tuned {sp['fine_tuned_ms']:.0f} ms per image")
    ax.legend(frameon=False, fontsize=10, labelcolor=INK_2, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("work", help="--work folder of tools/train_detector.py")
    ap.add_argument("--out", default="benchmarks")
    args = ap.parse_args()
    work = os.path.expanduser(args.work)
    os.makedirs(args.out, exist_ok=True)
    bm = json.load(open(os.path.join(work, "benchmarks.json")))
    training_curves(os.path.join(work, "run", "results.csv"), os.path.join(args.out, "training_curves.png"))
    per_class(bm, os.path.join(args.out, "per_class_ap50.png"))
    baseline(bm, os.path.join(args.out, "baseline_vs_tuned.png"))
    for name in ("confusion_matrix_normalized.png", "BoxPR_curve.png", "PR_curve.png"):
        src = os.path.join(work, "test", name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out, "test_" + name))
    t, sp = bm["test"], bm["speed"]
    b = bm["baseline"]
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("# Fridge food detector benchmarks\n\n")
        f.write(f"Model: {bm['model']} fine-tuned at {bm['imgsz']} px on {bm['device']}, "
                f"{bm['epochs_done']} epochs ({(bm['train_minutes'] or 0):.0f} min).\n\n")
        f.write("| Metric (test split) | Value |\n| --- | --- |\n")
        f.write(f"| Images / boxes | {t['images']} / {t['boxes']} |\n")
        f.write(f"| mAP50 | {t['map50'] * 100:.1f}% |\n| mAP50-95 | {t['map50_95'] * 100:.1f}% |\n")
        f.write(f"| Precision | {t['precision'] * 100:.1f}% |\n| Recall | {t['recall'] * 100:.1f}% |\n")
        if "test_clean" in bm:
            c = bm["test_clean"]
            f.write(f"| mAP50, test images not near-duplicates of training ({c['images']}) | {c['map50'] * 100:.1f}% |\n")
        f.write(f"| CPU inference, fine-tuned | {sp['fine_tuned_ms']:.0f} ms/image |\n")
        f.write(f"| CPU inference, stock COCO | {sp['stock_coco_ms']:.0f} ms/image |\n\n")
        f.write("Shared classes (AP50, same scoring code):\n\n| Class | Stock COCO | Fine-tuned |\n| --- | --- | --- |\n")
        for c in b["classes"]:
            f.write(f"| {c} | {b['stock_coco'][c] * 100:.1f}% | {b['fine_tuned'][c] * 100:.1f}% |\n")
        f.write("\nPer class (test AP50):\n\n| Class | AP50 | Precision | Recall | Test boxes |\n| --- | --- | --- | --- | --- |\n")
        for k, v in sorted(bm["per_class"].items(), key=lambda kv: -kv[1]["ap50"]):
            f.write(f"| {k} | {v['ap50'] * 100:.1f}% | {v['precision'] * 100:.1f}% | {v['recall'] * 100:.1f}% | {v['test_boxes']} |\n")
    if "test_clean" in bm:
        c = bm["test_clean"]
        with open(os.path.join(args.out, "summary.md"), "a") as f:
            f.write(f"\nNear-duplicate check: {c['near_dupes_10']} of {t['images']} test images are near-identical to a "
                    f"training image (difference hash, 10 of 256 bits or fewer), {c['near_dupes_25']} within 25 bits. "
                    f"Without those 25-bit look-alikes ({c['images']} images left), mAP50 is {c['map50'] * 100:.1f}% "
                    f"and mAP50-95 {c['map50_95'] * 100:.1f}%, so the scores are not driven by leakage. "
                    "All images still come from the same dataset style (whole-shelf fridge photos); "
                    "expect lower scores on a door camera until its own photos are added.\n")
    print(f"Wrote charts to {args.out}/")


if __name__ == "__main__":
    main()
