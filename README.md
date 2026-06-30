<div align="center">

# ✦ FreshGrade AI

### *Know if your fruits & vegetables are fresh — in seconds.*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://aistudio.google.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

> Upload a photo of any fruit or vegetable.  
> AI detects every item, reads color, texture & surface condition,  
> then scores each one **out of 10** with a label and a detailed note.

<br/>

![FreshGrade AI Demo](https://img.shields.io/badge/🍎%20Fresh-8.4%2F10-22C55E?style=flat-square) &nbsp;
![FreshGrade AI Demo](https://img.shields.io/badge/🍋%20Use%20Soon-5.1%2F10-F59E0B?style=flat-square) &nbsp;
![FreshGrade AI Demo](https://img.shields.io/badge/🥦%20Discard-2.3%2F10-EF4444?style=flat-square)

</div>

---

## 🧠 What It Does

- 🔍 **Detects** every unique fruit and vegetable type in a single photo
- 🎨 **Analyzes** color, texture, surface blemishes, mold, and rot
- 📊 **Scores** each item from **0.0 – 10.0**
- 🏷️ **Labels** each item as **Fresh**, **Use Soon**, or **Discard**
- 📝 **Explains** the score with a detailed human-readable note
- 📷 Supports **drag & drop**, **file browse**, and **live camera capture**

---

## ⚡ Dual-Engine Architecture

FreshGrade AI runs **two analysis engines** and picks the best one automatically — no configuration needed.

```
📷 Image Upload
      │
      ▼
┌─────────────────────────────────────────────────────┐
│   Is GEMINI_API_KEY set and valid?                  │
│                                                     │
│   YES ──▶  🌟 Gemini Vision (Primary)              │
│            Full-scene analysis in one pass          │
│            Detects all fruits and vegetables + scores freshness   │
│            holistically — best for rot & mold       │
│                                                     │
│   NO  ──▶  🏠 YOLOv8 + CLIP (Local Fallback)       │
│            Detects items with bounding boxes        │
│            Scores each crop zero-shot via CLIP      │
│            Runs 100% offline — no API needed        │
└─────────────────────────────────────────────────────┘
      │
      ▼
📊 Score + Label + Note  →  Rendered on the UI
```

| Priority | Engine | How It Works | Requires |
|:---:|---|---|---|
| **1 — Primary** | 🌟 **Google Gemini Vision** | Analyzes the full image in one pass. Identifies all fruits and vegetables and assesses freshness holistically. Most accurate for rot, mold, and unusual items. | `GEMINI_API_KEY` in `.env` |
| **2 — Fallback** | 🏠 **YOLOv8 + CLIP (local)** | Detects items using YOLOv8 bounding boxes, crops each one, then scores against fresh vs. rotten text prompts via CLIP. Runs fully offline. | Nothing — just Python |

> If Gemini quota is exhausted or the key is missing, the system **silently falls back** to local processing. No manual intervention needed.

---

## 🏷️ Freshness Labels

| Label | Score Range | What It Means |
|---|:---:|---|
| 🟢 **Fresh** | 7.5 – 10.0 | Vibrant color, firm appearance, no visible damage — safe to eat |
| 🟡 **Use Soon** | 4.5 – 7.4 | Noticeable aging, minor blemishes or soft spots — eat within 1–2 days |
| 🔴 **Discard** | 0.0 – 4.4 | Visible rot, mold, heavy bruising, or advanced spoilage — do not consume |

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| 🌟 Primary Analysis | Google Gemini Vision (`gemini-2.5-flash` → `gemini-2.0-flash` → `gemini-1.5-flash`) |
| 🎯 Object Detection (fallback) | YOLOv8 nano — [Ultralytics](https://ultralytics.com) |
| 🧠 Quality Scoring (fallback) | CLIP ViT-B/32 — [OpenAI](https://github.com/openai/CLIP), runs locally |
| 🖼️ Image Processing | OpenCV + Pillow |
| ⚙️ Backend | FastAPI + Uvicorn |
| 🌐 Frontend | HTML + Vanilla CSS + Vanilla JS |
| 🔥 ML Framework | PyTorch |
| 🔐 Config | `python-dotenv` |

---

## 📁 Project Structure

```
FreshGrade AI/
│
├── 📂 backend/
│   ├── __init__.py
│   ├── main.py              ← FastAPI app, /upload endpoint, CORS, static file serving
│   ├── gemini_analyzer.py   ← 🌟 Gemini Vision integration (primary engine)
│   ├── detector.py          ← 🎯 YOLOv8 detection, bounding boxes, crop logic
│   ├── quality.py           ← 🧠 CLIP-based freshness scoring (fallback engine)
│   └── scoring.py           ← Maps CLIP output → /10 score + label + note
│
├── 📂 frontend/
│   ├── index.html           ← Full-page UI: hero, upload zone, results, how-it-works
│   ├── style.css            ← Dark mode, glassmorphism design system
│   └── app.js               ← Fetch, drag-and-drop, camera capture, card rendering
│
├── 📂 uploads/              ← Temp storage (auto-cleaned after every request)
├── .env.example             ← Template — copy to .env and add your Gemini key
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**
- A free [Google Gemini API key](https://aistudio.google.com/) *(optional — app works fully offline without it)*

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/manika-03/FreshGrade-AI.git
cd FreshGrade-AI

# 2. Install dependencies
pip install -r requirements.txt
# Windows users:
py -3 -m pip install -r requirements.txt
```

> ⚠️ **PyTorch** — if `torch` fails to install, get the right command for your OS + CUDA version from [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

### Configuration *(optional)*

```bash
# 3. Copy the example env file
cp .env.example .env          # macOS / Linux
copy .env.example .env        # Windows

# 4. Paste your Gemini API key into .env
#    GEMINI_API_KEY=AIza...
#    Get a free key → https://aistudio.google.com/
```

> If you skip this step, the app uses the local YOLOv8 + CLIP pipeline automatically.

### Run

```bash
# 5. Start the server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
# Windows:
py -3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 6. Open your browser
#    http://localhost:8000
```

---

## 🌐 API Reference

### `GET /health`

```json
{ "status": "ok", "message": "FreshGrade AI backend is running" }
```

---

### `POST /upload`

Accepts a multipart image. Returns one result object per unique fruit or vegetable type detected.

**Request**
```
Content-Type: multipart/form-data
Body: file = <image>    (PNG, JPG, WEBP, GIF, BMP — up to 20 MB)
```

**Response — Gemini engine** ✦
```json
{
  "engine": "gemini",
  "results": [
    {
      "item":  "Tomato",
      "score": 3.5,
      "label": "Discard",
      "note":  "Heavy rot on one side, dark mushy patches, significant mold present."
    },
    {
      "item":  "Banana",
      "score": 8.2,
      "label": "Fresh",
      "note":  "Bright yellow peel with minor speckling. Firm and ripe."
    }
  ]
}
```

**Response — Local fallback engine** 🏠
```json
{
  "engine": "local",
  "results": [
    {
      "item":  "Apple",
      "score": 7.1,
      "label": "Use Soon",
      "note":  "Slight surface bruising on one side. Still edible."
    }
  ],
  "gemini_error": "quota exceeded"
}
```

---

## 🎨 Frontend Features

| Feature | Detail |
|---|---|
| 🖱️ Drag & Drop | Drop an image directly onto the upload zone |
| 📂 File Browse | Click to open the OS file picker |
| 📷 Camera Capture | Opens device camera via `getUserMedia` and snaps a photo |
| 🖼️ Image Preview | Shows a thumbnail before analysis begins |
| ⏳ Animated Loader | Step-by-step status overlay while the backend processes |
| 🏷️ Engine Badge | Shows `Gemini ✦` or `Local` badge on results |
| ⚠️ Fallback Banner | Warning shown when running in local YOLOv8 + CLIP mode |
| 📊 Summary Bar | Color-coded count of Fresh / Use Soon / Discard items |
| 🃏 Result Cards | One card per item — name, score pill, label, and detailed note |
| 🔄 New Scan | One-click reset to analyze another image |

---

## ⚠️ Known Limitations

| Limitation | Detail |
|---|---|
| 📉 Gemini quota | Free-tier keys have rate limits. On 429 errors, the system falls back to local automatically. |
| 🎯 CLIP accuracy | Zero-shot CLIP scoring is less precise than a fine-tuned model. Visually similar items (e.g., red apple vs. tomato) may be misclassified by YOLOv8 in fallback mode. |
| 📦 Overlapping items | Heavily overlapping fruits and vegetables may count as a single detection in fallback mode. |
| 💡 Lighting | Works best with clear, well-lit images. Dark or blurry photos reduce accuracy. |
| 🌍 Rare items | Uncommon fruits and vegetables may not be detected by YOLOv8 (trained on COCO classes). |

---

## 🤖 Model Notes

**YOLOv8 nano (`yolov8n.pt`)** — Lightweight COCO-pretrained detection model (~6 MB). Downloaded automatically on first run. For better fruits and vegetables coverage, fine-tune on [Fruits-360](https://www.kaggle.com/datasets/moltean/fruits) (Kaggle, free).

**CLIP (`ViT-B/32`)** — OpenAI's zero-shot vision-language model. Downloaded automatically on first run (~350 MB). No fine-tuning required.

**Gemini Vision** — Tried in order: `gemini-2.5-flash` → `gemini-2.0-flash` → `gemini-1.5-flash`. Uses structured JSON output for reliable parsing. Requires a valid key from [Google AI Studio](https://aistudio.google.com/).

---

<div align="center">

**✦ FreshGrade AI** — AI-powered freshness grading for fruits and vegetables.

*Built with FastAPI · Gemini Vision · YOLOv8 · CLIP · PyTorch*

</div>
