"""
quality.py
──────────
Uses OpenAI CLIP (runs locally, no API key needed) to score
the freshness of each detected fruit or vegetable item.

Improved approach — multi-prompt ensemble:
  Instead of a binary softmax between 1 fresh prompt and 1 rotten prompt
  (which always gives ~50/50), we compare the image against a richer set of
  freshness-level prompts and compute a weighted freshness score.

  Freshness levels (5-way):
    1. Perfectly fresh  (score contribution: 1.0)
    2. Fresh            (score contribution: 0.80)
    3. Borderline       (score contribution: 0.50)
    4. Starting to spoil (score contribution: 0.20)
    5. Rotten / spoiled  (score contribution: 0.0)

  The final fresh_score is the weighted average of softmax probabilities.
  This gives a continuous and much more discriminative signal than binary.

Input:  list of detection dicts from detector.py
Output: list of {name, crop_path, fresh_score, rotten_score}
"""

import os
from PIL import Image

# ── Lazy imports ──────────────────────────────────────────────────────────────
try:
    import torch
    import clip
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False

_model   = None
_preproc = None
_device  = None
_classify_text_features = None
_text_feature_cache = {}


def _get_clip():
    global _model, _preproc, _device
    if _model is None:
        if not _CLIP_AVAILABLE:
            raise RuntimeError(
                "CLIP is not installed. Run: pip install openai-clip"
            )
        _device  = "cuda" if torch.cuda.is_available() else "cpu"
        _model, _preproc = clip.load("ViT-B/32", device=_device)
        _model.eval()
    return _model, _preproc, _device


# Freshness level weights: index 0 = best → index 4 = worst
FRESHNESS_WEIGHTS = [1.0, 0.80, 0.50, 0.20, 0.0]

# Detailed item-specific prompts for high-accuracy zero-shot evaluation
PRODUCE_PROMPTS = {
    "apple": [
        "a perfectly fresh, shiny, firm, crisp apple with vibrant color and no blemishes or bruises",
        "a fresh ripe apple in good condition with minor natural color variations",
        "a slightly aging apple with small soft spots or minor skin wrinkles",
        "an overripe apple with large brown bruises, soft spots, or significant shriveling",
        "a rotten decaying apple, completely mushy, brown, and covered in mold"
    ],
    "banana": [
        "a perfectly fresh, bright yellow ripe banana with green tips and no spots",
        "a fresh yellow banana in great condition with very few tiny brown spots",
        "a ripe banana with several brown spots, slightly soft but perfect for eating",
        "an overripe banana, heavily bruised, black, and very soft or shriveled",
        "a rotten black banana, completely decayed, moldy, and mushy"
    ],
    "orange": [
        "a perfectly fresh, firm, plump orange with vibrant orange skin and no blemishes",
        "a fresh ripe orange with smooth skin and good color",
        "a slightly soft orange with minor skin discoloration or blemishes",
        "an overripe orange with very soft spots, shriveled skin, or early white mold",
        "a rotten decaying orange, covered in green or white mold and completely spoiled"
    ],
    "broccoli": [
        "a perfectly fresh, compact, vibrant deep green broccoli head with firm stems",
        "a fresh green broccoli head in good condition",
        "a slightly limp broccoli head starting to show minor yellowing buds",
        "a wilting broccoli head with prominent yellow buds and soft or brown spots",
        "a rotten moldy broccoli head, dark brown or black, decaying and mushy"
    ],
    "carrot": [
        "a perfectly fresh, firm, crisp, bright orange carrot with clean smooth skin",
        "a fresh orange carrot in good condition",
        "a slightly limp or dry carrot with minor surface discoloration",
        "a bendable, shriveled carrot with dark ends, soft spots, or cracks",
        "a rotten decaying carrot, covered in mold, slimy, and black"
    ],
    "tomato": [
        "a perfectly fresh, plump, firm, shiny red tomato with no blemishes or cracks",
        "a fresh red tomato in good condition with smooth skin",
        "a slightly soft tomato with minor skin blemishes, spots, or light wrinkles",
        "an overripe mushy tomato with large dark bruises, cracks, or leaking fluid",
        "a rotten decaying tomato with visible mold, black rot lesions, and advanced decay"
    ],
    "potato": [
        "a perfectly firm, clean potato with smooth skin and no sprouts or greening",
        "a fresh potato in good condition",
        "a slightly soft potato with minor blemishes or tiny sprouts starting",
        "a shriveled, green, soft potato with prominent sprouts or dark internal bruises",
        "a rotten decaying potato, mushy, smelling bad, covered in mold and leaking fluid"
    ],
    "onion": [
        "a perfectly firm onion with dry, papery, clean skins and no sprouts",
        "a fresh onion in good condition with firm layers",
        "a slightly soft onion or one with minor skin blemishes",
        "a soft, shriveled onion with green sprouts or dark mold on outer skins",
        "a rotten decaying onion, completely soft, mushy, black, and moldy"
    ],
    "lemon": [
        "a perfectly fresh, firm, bright yellow lemon with smooth skin and no blemishes",
        "a fresh ripe lemon with good color",
        "a slightly soft lemon with minor skin spots or green patches",
        "a shriveled, soft lemon with dark spots or early signs of white mold",
        "a rotten decaying lemon, completely soft, covered in green or white mold"
    ],
    "lime": [
        "a perfectly fresh, firm, vibrant green lime with smooth skin and no blemishes",
        "a fresh ripe lime with good color",
        "a slightly soft lime with minor skin spots or yellow patches",
        "a shriveled, soft lime with dark spots or early signs of white mold",
        "a rotten decaying lime, completely soft, covered in green or white mold"
    ],
    "mango": [
        "a perfectly fresh, firm, vibrant ripe mango with smooth skin and no blemishes",
        "a fresh ripe mango with good color and slight softness",
        "a slightly soft mango with minor skin wrinkles or small dark spots",
        "an overripe mushy mango with large black bruises, deep wrinkles, or sap leakage",
        "a rotten decaying mango, completely mushy, black, moldy, and spoiled"
    ],
    "strawberry": [
        "a perfectly fresh, firm, bright red strawberry with fresh green leaves and no bruises",
        "a fresh red strawberry in good condition",
        "a slightly soft strawberry with minor bruising or light color",
        "a mushy overripe strawberry with dark bruises, leaking juice, or early white fuzz",
        "a rotten decaying strawberry, completely covered in grey mold and mushy"
    ],
    "avocado": [
        "a perfectly fresh, firm-yielding, dark green avocado with no bruises or hollow spots",
        "a fresh ripe avocado in good condition",
        "a soft avocado with minor skin indentations, slightly overripe but good",
        "an overripe, very mushy avocado with large sunken spots and brown flesh inside",
        "a rotten decaying avocado, completely mushy, rancid, moldy, and black"
    ],
    "mushroom": [
        "a perfectly fresh, firm, clean mushroom with a dry surface and tight gills",
        "a fresh mushroom in good condition",
        "a slightly soft or bruised mushroom with minor brown discoloration",
        "a shriveled, slimy, dark brown mushroom with open gills and soft texture",
        "a rotten decaying mushroom, black, slimy, moldy, and completely decomposed"
    ],
    "pepper": [
        "a perfectly fresh, firm, shiny bell pepper with vibrant color and a green stem",
        "a fresh bell pepper in good condition with smooth skin",
        "a slightly soft bell pepper with minor wrinkles or skin blemishes",
        "a shriveled, soft bell pepper with large dark spots, deep wrinkles, or moldy stem",
        "a rotten decaying bell pepper, completely mushy, moldy, and collapsed"
    ]
}

def _build_prompts(item_name: str) -> list[str]:
    """Build a 5-level freshness prompt set for the given fruit or vegetable item."""
    n = item_name.lower().strip()
    if n in PRODUCE_PROMPTS:
        return PRODUCE_PROMPTS[n]
    
    return [
        # Level 0 — Perfectly fresh
        f"a perfectly fresh {n} with vibrant color, firm texture, and no blemishes",
        # Level 1 — Fresh
        f"a fresh {n} with good color and no visible damage",
        # Level 2 — Borderline / use soon
        f"a slightly aging {n} with minor blemishes or soft spots",
        # Level 3 — Starting to spoil
        f"an overripe {n} with visible browning, bruising, or shriveling",
        # Level 4 — Rotten
        f"a rotten spoiled {n} with mold, heavy decay, and bad discoloration",
    ]


# Common fruits and vegetables candidates for local CLIP classification correction
PRODUCE_CLASSES = [
    "Apple", "Banana", "Orange", "Broccoli", "Carrot", "Grape", "Pear", 
    "Strawberry", "Lemon", "Lime", "Mango", "Tomato", "Potato", "Pepper", 
    "Cucumber", "Onion", "Garlic", "Cabbage", "Zucchini", "Eggplant", 
    "Avocado", "Peach", "Mushroom", "Chili"
]

NOISE_CLASSES = [
    "sports ball", "bowl", "cup", "potted plant", "vase", "sandwich",
    "hot dog", "pizza", "donut", "cake", "person", "hand", "background",
    "kitchen counter", "table"
]

CLASSIFY_CANDIDATES = PRODUCE_CLASSES + NOISE_CLASSES


def score_items(detections: list[dict]) -> list[dict]:
    """
    For each detected item, run CLIP multi-prompt freshness scoring.

    Args:
        detections: list of dicts with keys: name, crop_path, bbox, confidence

    Returns:
        list of dicts: {name, crop_path, fresh_score, rotten_score}
    """
    if not detections:
        return []

    model, preproc, device = _get_clip()

    # ── Pre-encode ensembled classification candidates once to optimize performance (Cached globally) ──
    global _classify_text_features
    if _classify_text_features is None:
        try:
            candidate_vectors = []
            for c in CLASSIFY_CANDIDATES:
                prompts = [
                    f"a photo of a {c.lower()}",
                    f"a {c.lower()} fruit or vegetable",
                    f"a close-up of a {c.lower()}",
                    f"a fresh or rotten {c.lower()}",
                    f"a crop of a {c.lower()}"
                ]
                tokens = clip.tokenize(prompts, truncate=True).to(device)
                with torch.no_grad():
                    features = model.encode_text(tokens)
                    features = features / features.norm(dim=-1, keepdim=True)
                    avg_feature = features.mean(dim=0)
                    avg_feature = avg_feature / avg_feature.norm(dim=-1)
                    candidate_vectors.append(avg_feature)
            _classify_text_features = torch.stack(candidate_vectors)
        except Exception as e:
            print(f"Failed to pre-encode classification prompts: {e}")

    classify_text_features = _classify_text_features

    # ── 1. Load and preprocess all crops ──
    valid_dets = []
    img_tensors = []

    for det in detections:
        crop_path = det["crop_path"]
        if not os.path.exists(crop_path):
            continue
        try:
            pil_img = Image.open(crop_path).convert("RGB")
            tensor = preproc(pil_img)
            img_tensors.append(tensor)
            valid_dets.append(det)
        except Exception as e:
            print(f"Failed to load or preprocess crop {crop_path}: {e}")
            continue

    if not img_tensors:
        return []

    # ── 2. Run batched image encoding ──
    try:
        batch_tensor = torch.stack(img_tensors).to(device)
        with torch.no_grad():
            batch_features = model.encode_image(batch_tensor)
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
    except Exception as e:
        print(f"Batched CLIP image encoding failed: {e}")
        return []

    scored = []

    # ── 3. Process each item using its pre-encoded feature vector ──
    for det, feat in zip(valid_dets, batch_features):
        item_name = det["name"]
        crop_path = det["crop_path"]
        image_features = feat.unsqueeze(0)  # shape [1, 512] to match original logic

        with torch.no_grad():
            # ── 1. Correct fruit or vegetable name using CLIP classification ──
            if classify_text_features is not None:
                logit_scale = model.logit_scale.exp().clamp(max=100)
                classify_logits = (logit_scale * image_features @ classify_text_features.T).squeeze(0)
                classify_probs = classify_logits.softmax(dim=0).cpu().tolist()
                best_idx = classify_probs.index(max(classify_probs))
                best_class = CLASSIFY_CANDIDATES[best_idx]
                best_prob = classify_probs[best_idx]

                # If the item is classified as noise/background, discard it immediately
                if best_class in NOISE_CLASSES:
                    continue

                # If CLIP has reasonable confidence, correct the name (increased threshold to 0.20)
                if best_prob > 0.20:
                    item_name = best_class
                elif item_name.lower().startswith("candidate_"):
                    # If it was a generic candidate and CLIP classification is not confident, discard it
                    continue

            # ── 2. Build and encode prompts for the corrected item name (Cached globally) ──
            cache_key = item_name.lower().strip()
            if cache_key not in _text_feature_cache:
                prompts = _build_prompts(item_name)
                binary_prompts = [
                    f"a fresh, healthy, clean {item_name.lower()} in perfect condition",
                    f"a rotten, moldy, spoiled, decayed, or damaged {item_name.lower()} with spots, holes, or bruises"
                ]
                try:
                    text_tokens = clip.tokenize(prompts, truncate=True).to(device)
                    binary_tokens = clip.tokenize(binary_prompts, truncate=True).to(device)
                    
                    text_feats = model.encode_text(text_tokens)
                    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                    
                    binary_feats = model.encode_text(binary_tokens)
                    binary_feats = binary_feats / binary_feats.norm(dim=-1, keepdim=True)
                    
                    _text_feature_cache[cache_key] = {
                        "freshness": text_feats,
                        "binary": binary_feats
                    }
                except Exception as e:
                    print(f"Failed to encode and cache prompts for {item_name}: {e}")
                    continue

            cached_features = _text_feature_cache.get(cache_key)
            if cached_features is None:
                continue

            text_features = cached_features["freshness"]
            binary_features = cached_features["binary"]

            # Cosine similarity logits (scaled for sharper softmax)
            logit_scale = model.logit_scale.exp().clamp(max=100)
            logits = (logit_scale * image_features @ text_features.T).squeeze(0)  # [5]
            probs  = logits.softmax(dim=0).cpu().tolist()  # [p0..p4]

            # ── 3. Run binary rot detection for penalty ──
            binary_logits = (logit_scale * image_features @ binary_features.T).squeeze(0)
            binary_probs = binary_logits.softmax(dim=0).cpu().tolist()
            rotten_prob = binary_probs[1]

        # ── Weighted freshness score in [0, 1] ──
        fresh_score_5level  = sum(p * w for p, w in zip(probs, FRESHNESS_WEIGHTS))
        
        # Apply calibrated quadratic rot penalty (soft curve to prevent harsh drops for minor spots)
        rot_penalty_multiplier = 1.0 - (rotten_prob ** 2)
        fresh_score = fresh_score_5level * rot_penalty_multiplier
        rotten_score = 1.0 - fresh_score

        scored.append({
            "name":         item_name,
            "crop_path":    crop_path,
            "confidence":   det.get("confidence", 1.0),
            "fresh_score":  fresh_score,
            "rotten_score": rotten_score,
            # Expose raw probs for debugging
            "level_probs":  [round(p, 4) for p in probs],
        })

    return scored
