"""
gemini_analyzer.py
──────────────────
Uses Google Gemini Vision (gemini-2.0-flash) to analyze the FULL image and:
  1. Identify every unique fruit/vegetable present
  2. Assess freshness for each one by looking at the entire scene

This replaces the broken YOLOv8n + CLIP pipeline which:
  - Misidentifies fruits and vegetables (tomato → apple because both are round and red in COCO)
  - Only looks at small cropped regions, missing context like rot, mold, blemishes
  - CLIP zero-shot scoring is unreliable for real-world rot detection

Requires:
  - GEMINI_API_KEY environment variable (or .env file)
  - pip install google-genai python-dotenv
"""

import os
import json
import re
import io
from pathlib import Path
from PIL import Image
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

_client = None


def _load_api_key() -> str:
    """Load the Gemini API key from env or .env file."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or api_key.startswith("YOUR_"):
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    return api_key


def _get_client():
    global _client
    if _client is None:
        if not _GEMINI_AVAILABLE:
            raise RuntimeError(
                "google-genai not installed. Run: pip install google-genai"
            )
        api_key = _load_api_key()
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Add it to your environment or create a .env file "
                "in the project root with: GEMINI_API_KEY=your_key_here"
            )
        _client = genai.Client(api_key=api_key)
    return _client


# ─── Schema for Structured Output ─────────────────────────────────────────────
class FruitOrVegetableItem(BaseModel):
    item: str = Field(description="Unique name of the fruit or vegetable type, e.g. Tomato, Banana, Apple.")
    score: float = Field(description="Freshness score on a scale of 0.0 to 10.0.")
    label: str = Field(description="Label matching the score: 'Fresh' (if score >= 7.5), 'Use Soon' (if between 4.5 and 7.4), or 'Discard' (if < 4.5).")
    note: str = Field(description="Detailed description of the condition, color, blemishes, mold, rot, or freshness reasons.")
    reasoning: str = Field(description="Step-by-step reasoning explaining the score based on visual features.")

# ─── Prompt ──────────────────────────────────────────────────────────────────
ANALYSIS_PROMPT = """
You are an expert food quality analyst. Analyze this image of fruits and/or vegetables.

Your task:
1. Identify every UNIQUE type of fruit or vegetable visible in the image.
   - If there are multiple of the same type (e.g., 2 tomatoes), list that type only ONCE.
   - Focus on actual fruits and vegetables only (ignore hands, bowls, bags, etc.)
2. For each unique fruit or vegetable type, carefully assess its freshness by examining:
   - Color: Is it vibrant and typical, or discolored/brown/yellow?
   - Surface: Are there blemishes, dark spots, mold, cracks, holes, or decay?
   - Texture: Does it look firm and fresh, or shriveled/wrinkled/mushy?
   - Overall condition: Any signs of rot, disease, or spoilage?

IMPORTANT: Be ACCURATE and HONEST. If something clearly has rot, mold, dark spots,
holes, or heavy damage — score it LOW (Discard range). Do NOT give high scores
to obviously damaged items. A rotten tomato should NEVER get a Fresh score.

Score each item on a scale of 0–10:
  9.0–10.0 → Fresh (excellent condition, no visible damage, vibrant color)
  7.5–8.9  → Fresh (good condition, minor surface variation only)
  4.5–7.4  → Use Soon (noticeable aging, minor blemishes, soft spots starting)
  2.0–4.4  → Discard (significant browning, bruising, clear decay)
  0.0–1.9  → Discard (heavy rot, mold, advanced spoilage — dangerous)

Label rules (MUST match score):
  score >= 7.5 → label = "Fresh"
  score >= 4.5 → label = "Use Soon"
  score <  4.5 → label = "Discard"
"""


def analyze_image(image_path: str) -> list[dict]:
    """
    Send the full image to Gemini Vision and get freshness analysis.

    Args:
        image_path: Absolute path to the image file.

    Returns:
        list of dicts: [{item, score, label, note}, ...]
        Compatible with the existing API response schema.
    """
    client = _get_client()

    # Downscale and compress image to reduce upload time & API latency
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Check dimensions and resize if larger than 1024px in any dimension
            max_dim = 1024
            width, height = img.size
            if width > max_dim or height > max_dim:
                if width > height:
                    new_width = max_dim
                    new_height = int(height * (max_dim / width))
                else:
                    new_height = max_dim
                    new_width = int(width * (max_dim / height))
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save to JPEG bytes in memory with quality=85
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=85)
            image_bytes = img_byte_arr.getvalue()
            mime_type = "image/jpeg"
    except Exception as e:
        print(f"Image resizing failed, using original file: {e}")
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        # Detect MIME type
        suffix = Path(image_path).suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp",
            ".gif": "image/gif", ".bmp": "image/jpeg",
        }
        mime_type = mime_map.get(suffix, "image/jpeg")

    # Build the multimodal request using new SDK with Structured JSON outputs
    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]
    response = None
    last_error = None

    for model_name in models_to_try:
        try:
            print(f"Attempting analysis with {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    ANALYSIS_PROMPT,
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                    response_schema=list[FruitOrVegetableItem],
                ),
            )
            print(f"Successfully analyzed image using {model_name}.")
            break
        except Exception as e:
            print(f"Model {model_name} failed: {e}")
            last_error = e

    if response is None:
        raise last_error or RuntimeError("All Gemini models failed and no error was captured.")

    # Parse response.text as JSON directly (assured by schema)
    raw_text = response.text.strip()
    try:
        items = json.loads(raw_text)
    except json.JSONDecodeError:
        # Attempt to recover a valid JSON array from partial/truncated response
        # Strategy: find the outermost array, then strip any trailing incomplete object
        match = re.search(r"\[.*", raw_text, re.DOTALL)
        if match:
            partial = match.group(0)
            # Try as-is
            try:
                items = json.loads(partial)
            except json.JSONDecodeError:
                # Strip incomplete last object and close the array
                last_complete = partial.rfind("},")
                if last_complete != -1:
                    trimmed = partial[:last_complete + 1] + "]"
                    try:
                        items = json.loads(trimmed)
                    except json.JSONDecodeError:
                        items = []
                else:
                    items = []
        else:
            items = []

    # Validate and sanitize each item
    results = []
    seen_names = set()
    for item in items:
        name = str(item.get("item", "Unknown")).strip()
        name_key = name.lower()

        # Deduplicate
        if name_key in seen_names:
            continue
        seen_names.add(name_key)

        raw_score = float(item.get("score", 5.0))
        score = round(max(0.0, min(10.0, raw_score)), 1)

        # Re-enforce label from score (don't trust Gemini's label if score is inconsistent)
        if score >= 7.5:
            label = "Fresh"
        elif score >= 4.5:
            label = "Use Soon"
        else:
            label = "Discard"

        note = str(item.get("note", "")).strip()
        if not note:
            note = str(item.get("reasoning", "Analysis complete.")).strip()

        results.append({
            "item":  name,
            "score": score,
            "label": label,
            "note":  note,
        })

    return results


def is_available() -> bool:
    """Return True if Gemini API key is configured and library is installed."""
    if not _GEMINI_AVAILABLE:
        return False

    api_key = _load_api_key()

    def is_valid_key(key: str) -> bool:
        k = key.strip().strip('"').strip("'")
        PLACEHOLDERS = {"your_gemini_api_key_here", "your_key_here", ""}
        if not k or k.lower() in PLACEHOLDERS:
            return False
        if k.upper().startswith("YOUR_"):
            return False
        return True

    return is_valid_key(api_key)
