"""
detector.py
───────────
Uses YOLOv8 (Ultralytics) to detect fruits and vegetables in an image.

Returns a list of detection dicts (one per UNIQUE item type):
  {
    "name":      str,        # e.g. "apple"
    "crop_path": str,        # absolute path to the best (highest-confidence) crop
    "bbox":      [x1,y1,x2,y2],
    "confidence": float
  }

Deduplication: if multiple instances of the same item are detected (e.g. 2
broccoli heads), only the highest-confidence crop is kept — the app reports
item types, not counts.

Model: yolov8n.pt (COCO pretrained). COCO has only ~10 food-related classes;
we filter to those and map them to clean display names.
"""

import os
import uuid
from pathlib import Path
from PIL import Image

# ── Lazy import so FastAPI starts even if torch is not yet installed ──
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# COCO class names that are actual fruits / vegetables (YOLOv8 COCO has 80
# classes; only these are food fruits and vegetables).  Key = exact COCO label (lower),
# Value = clean display name.
# ─────────────────────────────────────────────────────────────────────────────
PRODUCE_MAP = {
    "apple":      "Apple",
    "banana":     "Banana",
    "orange":     "Orange",
    "broccoli":   "Broccoli",
    "carrot":     "Carrot",
    # Below are in some YOLOv8 variants trained on extended food datasets:
    "grape":      "Grape",
    "pear":       "Pear",
    "pineapple":  "Pineapple",
    "strawberry": "Strawberry",
    "watermelon": "Watermelon",
    "lemon":      "Lemon",
    "mango":      "Mango",
    "kiwi":       "Kiwi",
    "cherry":     "Cherry",
    "tomato":     "Tomato",
    "potato":     "Potato",
    "corn":       "Corn",
    "pepper":     "Pepper",
    "cucumber":   "Cucumber",
    "lettuce":    "Lettuce",
    "onion":      "Onion",
    "garlic":     "Garlic",
    "cabbage":    "Cabbage",
    "spinach":    "Spinach",
    "zucchini":   "Zucchini",
    "eggplant":   "Eggplant",
    "avocado":    "Avocado",
    "peach":      "Peach",
    "plum":       "Plum",
    "coconut":    "Coconut",
    "pomegranate":"Pomegranate",
    "papaya":     "Papaya",
    "grapefruit": "Grapefruit",
    "raspberry":  "Raspberry",
    "blueberry":  "Blueberry",
    "mushroom":   "Mushroom",
    "asparagus":  "Asparagus",
    "celery":     "Celery",
    "radish":     "Radish",
    "beet":       "Beet",
    "yam":        "Yam",
    "fig":        "Fig",
    "lime":       "Lime",
    "guava":      "Guava",
    "lychee":     "Lychee",
    "dragonfruit":"Dragonfruit",
    "jackfruit":  "Jackfruit",
    "durian":     "Durian",
    "tangerine":  "Tangerine",
    "cantaloupe": "Cantaloupe",
    "honeydew":   "Honeydew",
    "turnip":     "Turnip",
    "sweet potato": "Sweet Potato",
    "pumpkin":    "Pumpkin",
    "squash":     "Squash",
    "artichoke":  "Artichoke",
    "leek":       "Leek",
    "chili":      "Chili",
    "ginger":     "Ginger",
}

CROPS_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads", "crops")
os.makedirs(CROPS_DIR, exist_ok=True)

_model = None  # singleton — loaded once


def _get_model():
    global _model
    if _model is None:
        if not _YOLO_AVAILABLE:
            raise RuntimeError(
                "ultralytics is not installed. Run: pip install ultralytics"
            )
        model_path = os.path.join(os.path.dirname(__file__), "models", "yolov8n.pt")
        if os.path.exists(model_path):
            _model = YOLO(model_path)
        else:
            # Auto-download nano weights (≈6 MB) to project root
            root_model = os.path.join(os.path.dirname(__file__), "..", "yolov8n.pt")
            if os.path.exists(root_model):
                _model = YOLO(root_model)
            else:
                _model = YOLO("yolov8n.pt")
    return _model


# A list of auxiliary YOLO COCO classes that often represent or contain other fruits/vegetables.
CANDIDATE_COCO_CLASSES = {
    "sports ball", "potted plant", "bowl", "cup", "vase",
    "sandwich", "hot dog", "pizza", "donut", "cake"
}


def detect_items(image_path: str) -> list[dict]:
    """
    Run YOLOv8 on the given image.

    Strategy:
      1. Run inference with a low confidence threshold (0.25) to catch all fruits and vegetables.
      2. Keep fruits and vegetables classes from PRODUCE_MAP and auxiliary classes from CANDIDATE_COCO_CLASSES.
      3. Deduplicate standard fruits and vegetables classes (keep highest confidence). Keep all auxiliary detections
         as unique candidate items (e.g. Candidate_0_sports_ball) so CLIP can classify them.
      4. If NO fruits and vegetables are found, fall back to any detected food-like objects.

    Returns a list of detection dicts.
    """
    model  = _get_model()
    image  = Image.open(image_path).convert("RGB")
    width, height = image.size

    results = model(image_path, verbose=False, conf=0.25, iou=0.5)

    # best_per_type[label] = best detection dict so far
    best_per_type: dict[str, dict] = {}
    candidate_counter = 0

    for result in results:
        boxes = result.boxes
        names = result.names   # {class_id: class_name}

        for box in boxes:
            cls_id     = int(box.cls[0])
            confidence = float(box.conf[0])
            raw_label  = names.get(cls_id, "unknown").lower().strip()

            # Check if class is in fruits/vegetables map or auxiliary candidate list
            is_produce = raw_label in PRODUCE_MAP
            is_candidate = raw_label in CANDIDATE_COCO_CLASSES

            if not is_produce and not is_candidate:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(width, x2); y2 = min(height, y2)

            # Skip degenerate boxes
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue

            if is_produce:
                display_name = PRODUCE_MAP[raw_label]
                existing = best_per_type.get(display_name)
                if existing is None or confidence > existing["confidence"]:
                    best_per_type[display_name] = {
                        "name":       display_name,
                        "raw_label":  raw_label,
                        "bbox":       [x1, y1, x2, y2],
                        "confidence": round(confidence, 3),
                    }
            else:
                # For auxiliary candidates, keep them unique so we crop and classify them all
                unique_key = f"Candidate_{candidate_counter}_{raw_label.replace(' ', '_')}"
                candidate_counter += 1
                best_per_type[unique_key] = {
                    "name":       unique_key,
                    "raw_label":  raw_label,
                    "bbox":       [x1, y1, x2, y2],
                    "confidence": round(confidence, 3),
                }

    # Now crop the best representative for each detected type/candidate
    detections = []
    for key_name, det in best_per_type.items():
        x1, y1, x2, y2 = det["bbox"]
        crop = image.crop((x1, y1, x2, y2))
        crop_filename = f"{uuid.uuid4().hex}_{det['raw_label']}.jpg"
        crop_path = os.path.join(CROPS_DIR, crop_filename)
        crop.save(crop_path, "JPEG", quality=90)

        detections.append({
            "name":       key_name,
            "raw_label":  det["raw_label"],
            "crop_path":  crop_path,
            "bbox":       det["bbox"],
            "confidence": det["confidence"],
        })

    # ── Fallback: if nothing matched our fruits and vegetables/candidate list, use any high-confidence detection ──
    if not detections:
        for result in results:
            boxes = result.boxes
            names = result.names
            best_fallback: dict[str, dict] = {}

            for box in boxes:
                cls_id     = int(box.cls[0])
                confidence = float(box.conf[0])
                raw_label  = names.get(cls_id, "unknown").lower().strip()

                if confidence < 0.40:
                    continue

                existing = best_fallback.get(raw_label)
                if existing is None or confidence > existing["confidence"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(width, x2); y2 = min(height, y2)
                    best_fallback[raw_label] = {
                        "name":       raw_label.capitalize(),
                        "raw_label":  raw_label,
                        "bbox":       [x1, y1, x2, y2],
                        "confidence": round(confidence, 3),
                    }

            for raw_label, det in best_fallback.items():
                x1, y1, x2, y2 = det["bbox"]
                crop = image.crop((x1, y1, x2, y2))
                crop_filename = f"{uuid.uuid4().hex}_{raw_label}.jpg"
                crop_path = os.path.join(CROPS_DIR, crop_filename)
                crop.save(crop_path, "JPEG", quality=90)
                detections.append({
                    "name":       det["name"],
                    "raw_label":  raw_label,
                    "crop_path":  crop_path,
                    "bbox":       det["bbox"],
                    "confidence": det["confidence"],
                })

    return detections


def cleanup_crops(detections: list[dict]):
    """Remove temporary crop files."""
    for det in detections:
        cp = det.get("crop_path", "")
        if cp and os.path.exists(cp):
            try:
                os.remove(cp)
            except OSError:
                pass
