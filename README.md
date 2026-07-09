# Sugarcane Diseases and Pest Management using Deep Learning

A multimodal AI pipeline that detects **tiller borer infestation** and **yellow leaf disease** in sugarcane crops by combining computer vision (YOLOv8) with a tabular symptom-based classifier (TabNet), fusing both signals into a single final diagnosis.

Built for AgriThon 2.0 — 48-Hour Hackathon on "AI for Sugarcane Diseases and Pest Management," organized by the School of Computer Science and Information Systems, VIT Vellore, sponsored by the Department of Biotechnology, Govt. of India.

---

## Problem statement

Farmers often can't identify sugarcane pests and diseases early enough to act. This project combines two independent signals — what a photo shows, and what a farmer reports through simple yes/no symptom questions — into one reliable detection system for two of the most damaging sugarcane threats:

- **Tiller borer** — a stem-boring insect pest
- **Yellow leaf disease** — a viral/phytoplasma disease causing progressive leaf yellowing

---

## Dataset

| Class | Images collected | Annotation type |
|---|---|---|
| Tiller borer | 1,500+ | Bounding box |
| Yellow leaf disease | 2,000+ | Polygon segmentation |

All images were field-collected photographs of infected sugarcane stems and leaves at varying disease/infestation stages, angles, and lighting conditions.

---

## Pipeline overview

```
Input 1: Tiller image          Input 2: Yellow leaf image
        │                               │
        ▼                               ▼
   YOLOv8s (bbox)              YOLOv8s-seg (polygon)
        │                               │
        ▼                               ▼
   Output-1: Tiller            Output-2: Disease
   borer detected               region detected
        │                               │
Input 3: Farmer Q&A            Input 4: Farmer Q&A
(tiller symptoms)              (yellow leaf symptoms)
        │                               │
        ▼                               ▼
   TabNet classifier            TabNet classifier
        │                               │
        ▼                               ▼
   Output-3: Tiller             Output-4: Disease
   presence predicted           presence predicted
        │                               │
        └───────────────┬───────────────┘
                         ▼
                  Fusion layer
                         ▼
              Final output: presence/
              absence of tiller borer
              and yellow leaf disease
```

Two independent detection tracks — image-based and text-based — are fused per class to reduce false negatives from either signal alone.

---

## Step-by-step process

### 1. Raw data collection
1,500+ tiller borer images and 2,000+ yellow leaf disease images were collected directly from sugarcane fields, capturing multiple growth/infection stages, lighting conditions, and camera angles to build a realistic, varied dataset.

### 2. Preprocessing
All raw images (originally in mixed formats and resolutions — `.jpg`, `.png`, `.webp`, varying dimensions) were standardized before annotation:
- Resized to a uniform **640×640** resolution (matching YOLOv8's default training input size)
- Converted to a single consistent format (`.jpg`)
- Pixel-level normalization applied during training via the YOLOv8/Ultralytics pipeline

### 3. Annotation (CVAT)
Annotation was performed in CVAT, with two distinct annotation types matched to the nature of each class:

- **Tiller borer → Bounding box.** The pest is a discrete, localizable object, so a tightly-drawn rectangle around each visible larva/damage instance was sufficient.
- **Yellow leaf disease → Polygon segmentation.** The disease presents as an irregular region of discoloration across the leaf blade, so polygon tracing was used to capture the true diseased boundary rather than an approximate box.

Each annotated image was exported in YOLO-compatible label format:
- Bounding box: `class x_center y_center width height` (normalized 0–1)
- Polygon: `class x1 y1 x2 y2 x3 y3 ...` (normalized point pairs)

### 4. Augmentation
To improve model robustness and expand effective dataset size, augmentation was applied post-annotation using Albumentations, with labels transformed consistently alongside each image:
- Rotation (±25°)
- Horizontal flip
- Brightness/contrast adjustment
- Color jittering
- Gaussian noise

For the segmentation dataset, polygon labels were converted to pixel masks before augmentation (to correctly handle geometric transforms like rotation), then re-extracted as polygons afterward — preserving label accuracy through the transformation.

### 5. Model training

**Vision models:**
- `YOLOv8s` trained on the bounding-box-annotated tiller borer dataset for object detection
- `YOLOv8s-seg` trained on the polygon-annotated yellow leaf disease dataset for segmentation

Both were trained using the Ultralytics framework, with training/validation splits, tracked via standard object detection/segmentation metrics (mAP, precision, recall) logged per epoch.

**Tabular models:**
- Two separate `TabNet` classifiers were trained on synthetic symptom-based Q&A data (`synthetic_tiller_data.csv`, `yellow_leaf_synthetic_data.csv`) — Yes/No responses to visual symptom questions mapped to disease/pest presence labels.
- Scripts: `train_tabnet_tiller.py`, `train_tabnet_yld.py`

### 6. Compiling results
- Best-performing YOLO checkpoints (`best.pt`) saved from each training run based on validation mAP
- Trained TabNet models exported and zipped (`tabnet_model.zip`, `yellow_leaf_tabnet_model.zip`)
- Output metrics and sample detection screenshots documented in `Overall Output Screenshots.docx`

### 7. Fusion and final output
The fusion layer combines all four model outputs (YOLO detection/segmentation confidence + TabNet symptom-based prediction) per class to produce the final verdict:
- Tiller borer: present / not present
- Yellow leaf disease: present / not present

This dual-signal approach means a low-confidence image detection can still be confirmed (or contradicted) by the farmer's reported symptoms, and vice versa — improving reliability over either signal alone.

---

## Repository structure

```
├── README.md
├── Overall Output Screenshots.docx        # Sample outputs and results
├── create_yld_features_and_template.py    # Feature engineering for yellow leaf TabNet
├── synthetic_tiller_data.csv              # TabNet training data (tiller)
├── yellow_leaf_synthetic_data.csv         # TabNet training data (yellow leaf)
├── train_tabnet_tiller.py                 # TabNet training script (tiller)
├── train_tabnet_yld.py                    # TabNet training script (yellow leaf)
├── tabnet_model.zip                       # Trained TabNet weights (tiller)
├── yellow_leaf_tabnet_model.zip           # Trained TabNet weights (yellow leaf)
```

> Note: YOLOv8s / YOLOv8s-seg training scripts and annotated image datasets are maintained locally due to size and are being progressively added to this repository.

---

## Tech stack

- **YOLOv8s** — object detection (tiller borer)
- **YOLOv8s-seg** — instance segmentation (yellow leaf disease)
- **TabNet** — tabular deep learning classifier (symptom-based prediction)
- **CVAT** — image annotation
- **Albumentations** — data augmentation
- **Python, Ultralytics, PyTorch**

---

## Future work

- Complete YOLOv8s / YOLOv8s-seg upload with full training scripts and sample weights
- End-to-end inference script combining all four model outputs into the fusion layer
- Web/mobile interface for farmers to upload images and answer symptom questions
- Scale dataset further and benchmark model performance across more diverse field conditions

---

## Acknowledgements

Built for AgriThon 2.0, organized by the School of Computer Science and Information Systems, VIT Vellore, sponsored by the Department of Biotechnology, Government of India.
