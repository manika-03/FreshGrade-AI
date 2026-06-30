"""
scoring.py
──────────
Converts CLIP freshness scores into:
  - A score out of 10
  - A human-readable label: Fresh | Use Soon | Discard
  - A brief descriptive note

Score formula:
  fresh_score (weighted softmax output from quality.py) is in [0, 1].
  It is scaled to [0, 10]:
    1.0 → 10   (perfectly fresh)
    0.8 → 8    (fresh)
    0.5 → 5    (borderline — use soon)
    0.2 → 2    (starting to spoil)
    0.0 → 0    (rotten)

Label thresholds:
  7.5 – 10.0  → Fresh
  4.5 –  7.4  → Use Soon
  0.0 –  4.4  → Discard
"""

import os


# ── Note templates per label ───────────────────────────────────────────────────
NOTES = {
    "Fresh": [
        "Looks great! Good color and firm texture — no visible damage.",
        "Strong freshness indicators. Vibrant color and no blemishes detected.",
        "Excellent condition. Ready to eat or store normally.",
        "No signs of deterioration. Ideal freshness level detected.",
    ],
    "Use Soon": [
        "Minor blemishes or early discoloration detected. Still edible — use within 1–2 days.",
        "Slight surface irregularities noted. Consume soon for best quality.",
        "Early signs of aging detected. Best consumed in the next day or two.",
        "Some softening or discoloration present but still nutritionally sound. Use soon.",
    ],
    "Discard": [
        "Significant browning, bruising, or decay detected. Not recommended for consumption.",
        "Heavy surface damage or mold indicators observed. Should be discarded.",
        "Advanced spoilage signals detected — poor color and texture breakdown.",
        "Strong signs of rot or over-ripening. Discard to avoid health risk.",
    ],
}


def _pick_note(label: str, score: float) -> str:
    """Pick a note variant based on score position within the label band."""
    options = NOTES.get(label, ["Quality assessment complete."])
    # Use score decimal to consistently pick a variant
    idx = int((score % 1) * len(options)) % len(options)
    return options[idx]


def compute_grades(raw_scores: list[dict]) -> list[dict]:
    """
    Maps CLIP output → /10 score, label, and note.
    Deduplicates results by fruit or vegetable item, keeping the highest-scoring representative.

    Args:
        raw_scores: list of {name, crop_path, fresh_score, rotten_score, confidence}

    Returns:
        list of {item, score, label, note}   ← matches the API response schema
    """
    best_results = {}

    for entry in raw_scores:
        fresh_p    = entry.get("fresh_score", 0.5)
        confidence = entry.get("confidence", 1.0)

        # ── Map to /10 ──
        raw_score = fresh_p * 10.0

        # Slight confidence penalty for low YOLO confidence detections
        if confidence < 0.50:
            # Nudge score 10% toward the midpoint (5.0)
            raw_score = raw_score * 0.90 + 5.0 * 0.10

        score = round(max(0.0, min(10.0, raw_score)), 1)

        # ── Assign label ──
        if score >= 7.5:
            label = "Fresh"
        elif score >= 4.5:
            label = "Use Soon"
        else:
            label = "Discard"

        note = _pick_note(label, score)
        item_name = entry["name"]

        # Keep only the highest-scoring instance of each unique fruit or vegetable item
        existing = best_results.get(item_name)
        if existing is None or score > existing["score"]:
            best_results[item_name] = {
                "item":  item_name,
                "score": score,
                "label": label,
                "note":  note,
            }

        # ── Clean up crop file ──
        crop_path = entry.get("crop_path", "")
        if crop_path and os.path.exists(crop_path):
            try:
                os.remove(crop_path)
            except OSError:
                pass

    return list(best_results.values())
