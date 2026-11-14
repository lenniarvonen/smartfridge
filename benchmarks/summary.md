# Fridge food detector benchmarks

Model: yolo11n.pt fine-tuned at 320 px on cpu, 23 epochs (54 min).

| Metric (test split) | Value |
| --- | --- |
| Images / boxes | 561 / 4411 |
| mAP50 | 92.5% |
| mAP50-95 | 64.9% |
| Precision | 88.5% |
| Recall | 88.6% |
| mAP50, test images not near-duplicates of training (436) | 92.1% |
| CPU inference, fine-tuned | 33 ms/image |
| CPU inference, stock COCO | 32 ms/image |

Shared classes (AP50, same scoring code):

| Class | Stock COCO | Fine-tuned |
| --- | --- | --- |
| apple | 1.3% | 96.8% |
| banana | 22.3% | 99.7% |
| carrot | 8.8% | 96.2% |

Per class (test AP50):

| Class | AP50 | Precision | Recall | Test boxes |
| --- | --- | --- | --- | --- |
| strawberry | 99.5% | 100.0% | 99.6% | 348 |
| banana | 99.4% | 92.3% | 100.0% | 67 |
| butter | 99.1% | 98.1% | 96.5% | 260 |
| blueberry | 98.4% | 91.5% | 100.0% | 99 |
| shrimp | 98.4% | 98.0% | 85.1% | 302 |
| meat | 97.7% | 96.5% | 84.2% | 359 |
| tomato | 97.7% | 96.3% | 94.8% | 327 |
| mushroom | 97.6% | 79.9% | 98.8% | 84 |
| carrot | 97.4% | 85.9% | 96.0% | 253 |
| leafy_greens | 97.3% | 75.1% | 100.0% | 79 |
| milk | 96.8% | 98.1% | 93.8% | 429 |
| cream | 96.7% | 99.6% | 95.5% | 334 |
| apple | 96.6% | 80.9% | 95.5% | 246 |
| chicken | 96.0% | 83.3% | 92.9% | 224 |
| cheese | 95.6% | 96.1% | 92.9% | 255 |
| egg | 94.7% | 89.1% | 88.8% | 250 |
| bell_pepper | 91.6% | 92.6% | 82.4% | 17 |
| cabbage | 91.4% | 89.8% | 83.3% | 12 |
| green_beans | 91.1% | 88.3% | 85.8% | 162 |
| citrus | 86.1% | 88.7% | 79.9% | 249 |
| cucumber | 80.1% | 81.1% | 74.5% | 23 |
| eggplant | 35.9% | 45.6% | 28.1% | 32 |

Near-duplicate check: 98 of 561 test images are near-identical to a training image (difference hash, 10 of 256 bits or fewer), 125 within 25 bits. Without those 25-bit look-alikes (436 images left), mAP50 is 92.1% and mAP50-95 64.1%, so the scores are not driven by leakage. All images still come from the same dataset style (whole-shelf fridge photos); expect lower scores on a door camera until its own photos are added.
