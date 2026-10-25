# VinBigData_XAI - Full Stack Startup Guide

## Quick Start (Two Terminals Required)

### Terminal 1: Start FastAPI Backend

```powershell
python scripts/run_backend.py
```

**Expected Output:**
```
================================================================================
Starting VinBigData_XAI FastAPI Backend
================================================================================

API will be available at: http://127.0.0.1:8765
API Docs:                http://127.0.0.1:8765/docs

Press Ctrl+C to stop the server
```

### Terminal 2: Start Streamlit UI

```powershell
python scripts/run_ui.py
```

Or directly:
```powershell
streamlit run app/ui/app.py
```

**Expected Output:**
```
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

---

## What Each Service Does

| Service | Port | Purpose |
|---------|------|---------|
| **FastAPI Backend** | 8765 | Handles inference, XAI generation, image processing |
| **Streamlit UI** | 8501 | Web interface for uploading images and viewing results |

---

## Visual XAI Output

When you upload an X-ray image, the backend generates:

1. **Grad-CAM Overlay** - Heatmap showing which regions influenced the prediction
2. **Grad-CAM++ Overlay** - Refined attention heatmap with sharper localization
3. **Grad-CAM Attention Box** - Red bounding box highlighting key region
4. **Grad-CAM++ Attention Box** - Red bounding box from Grad-CAM++ method

All visualizations are **unique to each uploaded image** and update dynamically.

---

## Troubleshooting

### Error: "Connection refused" on port 8765
**Solution:** The FastAPI backend is not running. Start Terminal 1 first.

### Error: "ConnectionError" in UI
**Solution:** Wait 5-10 seconds for the backend to fully load before uploading images.

### Import Errors in Streamlit
**Solution:** Make sure you're in the correct directory and the virtual environment is activated:
```powershell
.\.venv\Scripts\Activate.ps1
```

### Model Checkpoint Not Found
**Solution:** Obtain the required checkpoint from a trusted source and
place it at the project-relative default path:
```
checkpoints\convnext_tiny\convnext_tiny_weak_classes_best.pth
```

If the checkpoint is stored elsewhere, set `VINBIGDATA_CHECKPOINT` in
`.env`. Relative paths are resolved from the project root. Intrinsic
prototype explanations use the separate DenseNet-backed checkpoint and
evidence manifest listed in `README.md`.

---

## API Documentation

Once the backend is running, view interactive API docs at:
```
http://127.0.0.1:8765/docs
```

### Key Endpoints

- `GET /` - Health check
- `POST /predict` - Get predictions for an image
- `POST /visual-xai/generate` - Generate Grad-CAM visualizations
- `GET /visual-xai/status` - Check available XAI artifacts

---

## Stopping Services

Press `Ctrl+C` in each terminal to gracefully stop the services.
