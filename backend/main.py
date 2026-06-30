from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import shutil
import uuid
import os

try:
    from backend.detector import detect_items, cleanup_crops
    from backend.quality import score_items
    from backend.scoring import compute_grades
    from backend import gemini_analyzer
except ImportError:
    from detector import detect_items, cleanup_crops
    from quality import score_items
    from scoring import compute_grades
    import gemini_analyzer


app = FastAPI(
    title="FreshGrade AI",
    description="AI-powered freshness grader for fruits and vegetables using YOLOv8 + CLIP",
    version="1.0.0"
)

# ── CORS — allow the HTML frontend to call this API ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # In production, restrict to your domain
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok", "message": "FreshGrade AI backend is running"}


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Accepts a multipart image.
    Returns a JSON array — one object per unique detected fruit/vegetable type.
    Each object: { item, score, label, note }
    """
    # ── 1. Validate file type ──
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted.")

    # ── 2. Save uploaded image to disk ──
    ext = os.path.splitext(file.filename or "upload.jpg")[1] or ".jpg"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    detections = []
    gemini_error = None
    try:
        # ── 3. Check if Gemini is available for high-fidelity grading ──
        if gemini_analyzer.is_available():
            try:
                results = gemini_analyzer.analyze_image(save_path)
                return JSONResponse(content={"engine": "gemini", "results": results}, status_code=200)
            except Exception as e:
                print(f"Gemini analysis failed: {str(e)}")
                gemini_error = str(e)

        # ── 4. Fallback: Local YOLOv8 + CLIP pipeline ──
        detections = detect_items(save_path)

        if not detections:
            return JSONResponse(content={"engine": "local", "results": [], "gemini_error": gemini_error}, status_code=200)

        # ── 5. Safety dedup: ensure no duplicate item names reach CLIP ──
        seen_names = set()
        unique_detections = []
        for det in detections:
            name_key = det["name"].lower()
            if name_key not in seen_names:
                seen_names.add(name_key)
                unique_detections.append(det)
        detections = unique_detections

        # ── 6. Score each unique crop with CLIP ──
        raw_scores = score_items(detections)

        # ── 7. Convert to /10 grade + label + note ──
        results = compute_grades(raw_scores)

        return JSONResponse(content={"engine": "local", "results": results, "gemini_error": gemini_error}, status_code=200)


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    finally:
        # ── 8. Clean up temp crop images ──
        try:
            cleanup_crops(detections)
        except Exception:
            pass
        # ── 8. Clean up uploaded image ──
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except Exception:
                pass


# ── Serve Frontend Static Files ──
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
