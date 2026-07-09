"""
Augments tiller (bounding box) and yellow_leaf (polygon segmentation) datasets.
Expects this folder structure:

dataset/
├── tiller/
│   ├── images/  (tiller_1.jpg ...)
│   └── labels/  (tiller_1.txt ...)   -> YOLO bbox format: class x_c y_c w h
└── yellow_leaf/
    ├── images/  (yellow_leaf_1.jpg ...)
    └── labels/  (yellow_leaf_1.txt ...) -> YOLO polygon format: class x1 y1 x2 y2 ...

Outputs augmented images + labels into:
dataset/tiller/augmented/images, dataset/tiller/augmented/labels
dataset/yellow_leaf/augmented/images, dataset/yellow_leaf/augmented/labels
"""

import os
import cv2
import numpy as np
import albumentations as A

NUM_AUGS_PER_IMAGE = 5  # how many augmented copies to generate per original image

# ---------- shared transform for bounding box (tiller) ----------
bbox_transform = A.Compose(
    [
        A.Rotate(limit=25, p=0.7, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.6),
        A.ColorJitter(p=0.5),
        A.GaussNoise(p=0.2),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
)

# ---------- shared transform for image+mask (yellow leaf) ----------
mask_transform = A.Compose(
    [
        A.Rotate(limit=25, p=0.7, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.6),
        A.ColorJitter(p=0.5),
        A.GaussNoise(p=0.2),
    ]
)


def read_bbox_labels(label_path):
    boxes, classes = [], []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
            classes.append(cls)
            boxes.append([x, y, w, h])
    return boxes, classes


def write_bbox_labels(label_path, boxes, classes):
    with open(label_path, "w") as f:
        for cls, box in zip(classes, boxes):
            f.write(f"{cls} {' '.join(f'{v:.6f}' for v in box)}\n")


def polygon_to_mask(label_path, img_w, img_h):
    """Rasterize each polygon in the label file onto a mask, one channel per instance."""
    polygons, classes = [], []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            pts = np.array(
                [[coords[i] * img_w, coords[i + 1] * img_h] for i in range(0, len(coords), 2)],
                dtype=np.int32,
            )
            polygons.append(pts)
            classes.append(cls)

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for idx, pts in enumerate(polygons, start=1):
        cv2.fillPoly(mask, [pts], color=idx)  # each instance gets a unique id
    return mask, classes


def mask_to_polygon_labels(mask, classes, img_w, img_h):
    """Convert an augmented mask back to YOLO polygon label lines."""
    lines = []
    for idx, cls in enumerate(classes, start=1):
        instance_mask = (mask == idx).astype(np.uint8)
        contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 3:
            continue
        norm_pts = []
        for pt in contour.reshape(-1, 2):
            norm_pts.append(pt[0] / img_w)
            norm_pts.append(pt[1] / img_h)
        lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in norm_pts))
    return lines


def augment_bbox_dataset(base_dir):
    img_dir = os.path.join(base_dir, "images")
    lbl_dir = os.path.join(base_dir, "labels")
    out_img_dir = os.path.join(base_dir, "augmented", "images")
    out_lbl_dir = os.path.join(base_dir, "augmented", "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(lbl_dir, stem + ".txt")
        if not os.path.exists(label_path):
            continue

        img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, fname)), cv2.COLOR_BGR2RGB)
        boxes, classes = read_bbox_labels(label_path)

        for i in range(NUM_AUGS_PER_IMAGE):
            result = bbox_transform(image=img, bboxes=boxes, class_labels=classes)
            aug_img = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
            aug_boxes = result["bboxes"]
            aug_classes = result["class_labels"]

            out_name = f"{stem}_aug{i}"
            cv2.imwrite(os.path.join(out_img_dir, out_name + ".jpg"), aug_img)
            write_bbox_labels(os.path.join(out_lbl_dir, out_name + ".txt"), aug_boxes, aug_classes)

        print(f"[tiller] {fname}: {NUM_AUGS_PER_IMAGE} augmented copies created")


def augment_polygon_dataset(base_dir):
    img_dir = os.path.join(base_dir, "images")
    lbl_dir = os.path.join(base_dir, "labels")
    out_img_dir = os.path.join(base_dir, "augmented", "images")
    out_lbl_dir = os.path.join(base_dir, "augmented", "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(lbl_dir, stem + ".txt")
        if not os.path.exists(label_path):
            continue

        img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, fname)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        mask, classes = polygon_to_mask(label_path, w, h)

        for i in range(NUM_AUGS_PER_IMAGE):
            result = mask_transform(image=img, mask=mask)
            aug_img = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
            aug_mask = result["mask"]

            lines = mask_to_polygon_labels(aug_mask, classes, w, h)
            if not lines:
                continue  # skip if polygon vanished after transform (e.g. rotated out of frame)

            out_name = f"{stem}_aug{i}"
            cv2.imwrite(os.path.join(out_img_dir, out_name + ".jpg"), aug_img)
            with open(os.path.join(out_lbl_dir, out_name + ".txt"), "w") as f:
                f.write("\n".join(lines) + "\n")

        print(f"[yellow_leaf] {fname}: {NUM_AUGS_PER_IMAGE} augmented copies created")


if __name__ == "__main__":
    DATASET_ROOT = "dataset"  # change if your dataset folder is elsewhere

    augment_bbox_dataset(os.path.join(DATASET_ROOT, "tiller"))
    augment_polygon_dataset(os.path.join(DATASET_ROOT, "yellow_leaf"))

    print("\nDone. Augmented data saved in:")
    print(f"  {DATASET_ROOT}/tiller/augmented/")
    print(f"  {DATASET_ROOT}/yellow_leaf/augmented/")