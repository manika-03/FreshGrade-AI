/* ═══════════════════════════════════════════════
   FreshGrade AI — app.js
   Handles: file upload, drag-drop, camera capture,
            fetch to backend, result card rendering
   ═══════════════════════════════════════════════ */

const BACKEND_URL = window.location.origin === 'null' || window.location.protocol === 'file:' 
  ? 'http://127.0.0.1:8000' 
  : window.location.origin;

// ── DOM refs ──────────────────────────────────────
const fileInput      = document.getElementById('fileInput');
const dropZone       = document.getElementById('dropZone');
const previewWrap    = document.getElementById('previewWrap');
const previewImg     = document.getElementById('previewImg');
const checkBtn       = document.getElementById('checkBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingStep    = document.getElementById('loadingStep');
const resultsSection = document.getElementById('resultsSection');
const cardsGrid      = document.getElementById('cardsGrid');
const itemCount      = document.getElementById('itemCount');
const summaryBar     = document.getElementById('summaryBar');

let selectedFile = null;
let cameraStream  = null;

// ── File Input Change ──────────────────────────────
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) setFile(file);
});

// ── Drag & Drop ────────────────────────────────────
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) setFile(file);
});

dropZone.addEventListener('click', (e) => {
  // Don't trigger if clicking buttons inside
  if (e.target.closest('button') || e.target.closest('.preview-wrap')) return;
  fileInput.click();
});

// ── Set File & Preview ─────────────────────────────
function setFile(file) {
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewWrap.style.display = 'block';

  // Hide the drop UI labels when showing preview
  dropZone.querySelector('.drop-icon').style.display = 'none';
  dropZone.querySelector('.drop-title').style.display = 'none';
  dropZone.querySelector('.drop-sub').style.display = 'none';
  dropZone.querySelector('.drop-buttons').style.display = 'none';

  checkBtn.disabled = false;
}

// ── Clear Image ────────────────────────────────────
function clearImage() {
  selectedFile = null;
  previewImg.src = '';
  previewWrap.style.display = 'none';
  fileInput.value = '';

  dropZone.querySelector('.drop-icon').style.display = '';
  dropZone.querySelector('.drop-title').style.display = '';
  dropZone.querySelector('.drop-sub').style.display = '';
  dropZone.querySelector('.drop-buttons').style.display = '';

  checkBtn.disabled = true;
  hideResults();
}

// ── Camera ────────────────────────────────────────
async function activateCamera() {
  const modal = document.getElementById('cameraModal');
  const video = document.getElementById('cameraVideo');
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
    video.srcObject = cameraStream;
    modal.style.display = 'flex';
  } catch (err) {
    alert('Camera not available: ' + err.message);
  }
}

function snapPhoto() {
  const video  = document.getElementById('cameraVideo');
  const canvas = document.getElementById('snapCanvas');
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  canvas.toBlob((blob) => {
    const file = new File([blob], 'camera-snap.jpg', { type: 'image/jpeg' });
    setFile(file);
    closeCamera();
  }, 'image/jpeg', 0.92);
}

function closeCamera() {
  const modal = document.getElementById('cameraModal');
  modal.style.display = 'none';
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
}

// ── Analyze ───────────────────────────────────────
async function analyzeImage() {
  if (!selectedFile) return;

  showLoading();
  hideResults();

  const formData = new FormData();
  formData.append('file', selectedFile);

  // Cycle loading step messages
  const steps = [
    'Preparing image for analysis…',
    'Detecting fruits, vegetables, and context…',
    'Running AI freshness grading…',
    'Assessing color, surface, and texture…',
    'Building your report…'
  ];
  let stepIndex = 0;
  const stepInterval = setInterval(() => {
    stepIndex = (stepIndex + 1) % steps.length;
    loadingStep.textContent = steps[stepIndex];
  }, 1800);

  try {
    const res = await fetch(`${BACKEND_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    clearInterval(stepInterval);

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Server error' }));
      hideLoading();
      showError(err.detail || 'Something went wrong.');
      return;
    }

    const data = await res.json();
    hideLoading();
    renderResults(data);

  } catch (err) {
    clearInterval(stepInterval);
    hideLoading();
    showError('Could not reach the backend. Make sure it is running on port 8000.');
  }
}

function renderResults(data) {
  let items = [];
  let engine = 'local';

  // Support both old array schema and new object schema
  if (data && Array.isArray(data.results)) {
    items = data.results;
    engine = data.engine || 'local';
  } else if (Array.isArray(data)) {
    items = data;
    engine = 'local';
  }

  if (!items || items.length === 0) {
    showError('No fruits or vegetables detected in this image. Try a clearer photo.');
    return;
  }

  // Update engine badge & fallback warning banner
  const engineBadge = document.getElementById('engineBadge');
  const fallbackBanner = document.getElementById('fallbackBanner');
  const errorDesc = document.querySelector('.fallback-error-desc');

  if (engineBadge) {
    engineBadge.style.display = 'inline-flex';
    if (engine === 'gemini') {
      engineBadge.className = 'engine-badge engine-gemini';
      engineBadge.innerHTML = '✨ Gemini AI Premium';
      if (fallbackBanner) fallbackBanner.style.display = 'none';
      if (errorDesc) errorDesc.style.display = 'none';
    } else {
      engineBadge.className = 'engine-badge engine-local';
      engineBadge.innerHTML = '🛡️ Local Fallback';
      if (fallbackBanner) {
        fallbackBanner.style.display = 'flex';
        if (errorDesc) {
          const geminiError = data.gemini_error || (data.results && data.results.gemini_error);
          if (geminiError) {
            errorDesc.innerHTML = `<strong>Diagnostic Error:</strong> ${escapeHtml(geminiError)}`;
            errorDesc.style.display = 'block';
          } else {
            errorDesc.style.display = 'none';
          }
        }
      }
    }
  }

  resultsSection.style.display = 'block';
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

  itemCount.textContent = items.length;
  const pluralEl = document.getElementById('itemPlural');
  if (pluralEl) pluralEl.textContent = items.length === 1 ? '' : 's';

  // Summary bar
  const counts = { Fresh: 0, 'Use Soon': 0, Discard: 0 };
  items.forEach(item => { if (counts[item.label] !== undefined) counts[item.label]++; });

  summaryBar.innerHTML = '';
  const chipClasses = { Fresh: 'chip-fresh', 'Use Soon': 'chip-soon', Discard: 'chip-discard' };
  const chipEmojis  = { Fresh: '✅', 'Use Soon': '⚠️', Discard: '❌' };

  Object.entries(counts).forEach(([label, count]) => {
    if (count === 0) return;
    const chip = document.createElement('div');
    chip.className = `summary-chip ${chipClasses[label]}`;
    chip.innerHTML = `<span>${chipEmojis[label]}</span> ${count} ${label}`;
    summaryBar.appendChild(chip);
  });

  // Cards
  cardsGrid.innerHTML = '';
  items.forEach((item, i) => {
    const card = buildCard(item, i);
    cardsGrid.appendChild(card);
    // Trigger score bar fill after a short delay for animation
    setTimeout(() => {
      const fill = card.querySelector('.score-bar-fill');
      if (fill) fill.style.width = `${(item.score / 10) * 100}%`;
    }, 100 + i * 80);
  });
}

function buildCard(item, index) {
  const labelClass = {
    'Fresh':    'card-fresh',
    'Use Soon': 'card-soon',
    'Discard':  'card-discard',
  }[item.label] || 'card-soon';

  const labelEmoji = {
    'Fresh':    '🟢',
    'Use Soon': '🟡',
    'Discard':  '🔴',
  }[item.label] || '🟡';

  const card = document.createElement('div');
  card.className = `result-card ${labelClass}`;
  card.style.animationDelay = `${index * 0.07}s`;

  card.innerHTML = `
    <div class="card-header">
      <div class="card-name">${escapeHtml(item.item)}</div>
      <div class="score-col">
        <div class="card-score">${Number(item.score).toFixed(1)}</div>
        <div class="card-score-sub">/ 10</div>
      </div>
    </div>

    <div class="score-bar-wrap">
      <div class="score-bar-fill" style="width:0%"></div>
    </div>

    <div class="card-label">${labelEmoji} ${escapeHtml(item.label)}</div>

    <p class="card-note">${escapeHtml(item.note)}</p>
  `;

  return card;
}

// ── Reset ─────────────────────────────────────────
function resetAll() {
  clearImage();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Helpers ───────────────────────────────────────
function showLoading() {
  loadingStep.textContent = 'Detecting items with YOLOv8…';
  loadingOverlay.style.display = 'flex';
}

function hideLoading() {
  loadingOverlay.style.display = 'none';
}

function hideResults() {
  resultsSection.style.display = 'none';
  cardsGrid.innerHTML = '';
  summaryBar.innerHTML = '';
  
  const engineBadge = document.getElementById('engineBadge');
  const fallbackBanner = document.getElementById('fallbackBanner');
  if (engineBadge) engineBadge.style.display = 'none';
  if (fallbackBanner) fallbackBanner.style.display = 'none';
}

function showError(message) {
  cardsGrid.innerHTML = `
    <div style="grid-column:1/-1; text-align:center; padding:40px 0; color:var(--clr-discard);">
      <div style="font-size:2.5rem; margin-bottom:12px;">⚠️</div>
      <p style="font-size:1rem;">${escapeHtml(message)}</p>
    </div>
  `;
  resultsSection.style.display = 'block';
  resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Navbar shrink on scroll ───────────────────────
window.addEventListener('scroll', () => {
  const navbar = document.getElementById('navbar');
  if (window.scrollY > 60) {
    navbar.style.padding = '10px 48px';
  } else {
    navbar.style.padding = '18px 48px';
  }
});

// ── Testing Auto-load (for automated verification) ──
if (new URLSearchParams(window.location.search).get('test') === 'true') {
  window.addEventListener('DOMContentLoaded', async () => {
    try {
      const response = await fetch('test_image.png');
      const blob = await response.blob();
      const file = new File([blob], 'test_image.png', { type: 'image/png' });
      setFile(file);
      // Wait 1.5s to let the subagent see the preview, then analyze
      setTimeout(() => {
        analyzeImage();
      }, 1500);
    } catch (err) {
      console.error('Test auto-load failed:', err);
    }
  });
}
