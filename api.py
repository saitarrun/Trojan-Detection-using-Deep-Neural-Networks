
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Security, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os
import shutil
import numpy as np
from torchvision import models, transforms
from PIL import Image
import io
import base64
import uuid
import logging
import secrets

from models import get_resnet18
from dataset import get_cifar10_dataloaders
from defenses import NeuralCleanse, STRIP, ActivationClustering, RiskFusionEngine, WeightAnalysis
from gradcam_utils import GradCAM
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    expected = os.environ.get("API_TOKEN", "dev-secret-do-not-use-in-prod")
    if api_key != expected:
        raise HTTPException(status_code=403, detail={"error": "invalid_api_key"})
    return api_key

app = FastAPI(title="Gemini Trojan Detection API", description="Enterprise MLOps API for auditing Deep Neural Networks for Trojans.")

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

app.add_middleware(SecurityHeadersMiddleware)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_cors_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScanResponse(BaseModel):
    status: str
    model_analyzed: str
    fusion_risk_score: float
    details: dict
    gradcam_heatmap_b64: str | None

def determine_risk_level(score: float) -> str:
    if score > 0.75:
        return "CRITICAL (Deployment Blocked)"
    elif score > 0.40:
        return "WARNING (Manual Review Required)"
    else:
        return "SAFE (Cleared for Production)"

class AsyncScanResponse(BaseModel):
    status: str
    task_id: str
    message: str

class LocalPathScanRequest(BaseModel):
    model_path: str
    target_class: int = Field(-1, ge=-1, le=9999)
    trigger_type: str = Field("checkerboard", min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$')

_MAGIC_BYTES = {
    ".pth": (b"PK", b"\x80"),
    ".pt":  (b"PK", b"\x80"),
    ".onnx": (b"\x08",),
}

def _validate_magic(path: str, ext: str) -> bool:
    allowed = _MAGIC_BYTES.get(ext, ())
    with open(path, "rb") as f:
        header = f.read(8)
    return any(header.startswith(m) for m in allowed)

@app.post("/api/v1/scan-local-path", response_model=AsyncScanResponse)
@limiter.limit("10/minute")
async def scan_local_path(request: Request, body: LocalPathScanRequest, _key: str = Depends(verify_api_key)):
    if not os.path.exists(body.model_path):
        raise HTTPException(status_code=404, detail="Model file not found on server.")

    uploads_dir = os.path.realpath("uploads")
    requested = os.path.realpath(body.model_path)
    if not requested.startswith(uploads_dir + os.sep):
        raise HTTPException(status_code=403, detail="Access to paths outside uploads/ is not permitted.")

    valid_extensions = (".pth", ".pt", ".onnx")
    if not any(body.model_path.endswith(ext) for ext in valid_extensions):
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {valid_extensions}")

    try:
        from celery_worker import run_model_scan_task
        task = run_model_scan_task.delay(body.model_path, body.target_class, body.trigger_type)
        return AsyncScanResponse(
            status="accepted",
            task_id=task.id,
            message=f"Local model {os.path.basename(body.model_path)} accepted for analysis."
        )
    except Exception:
        err_id = str(uuid.uuid4())
        logger.exception(f"[trace_id={err_id}] scan_local_path failed")
        raise HTTPException(status_code=500, detail=f"Internal error. trace_id={err_id}")

@app.post("/api/v1/scan-model", response_model=AsyncScanResponse)
@limiter.limit("10/minute")
async def scan_model_async(
    request: Request,
    model_file: UploadFile = File(...),
    target_class: int = Form(-1),
    trigger_type: str = Form("checkerboard"),
    _key: str = Depends(verify_api_key),
):
    valid_extensions = (".pth", ".pt", ".onnx")
    raw_name = model_file.filename or ""
    ext = os.path.splitext(raw_name)[1].lower()
    if ext not in valid_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {valid_extensions}")

    if not (-1 <= target_class <= 9999):
        raise HTTPException(status_code=422, detail="target_class must be -1 for auto-detect or 0-9999")
    if not trigger_type or len(trigger_type) > 64:
        raise HTTPException(status_code=422, detail="trigger_type must be 1-64 characters")

    try:
        os.makedirs("uploads", exist_ok=True)
        safe_name = secrets.token_hex(16) + ext
        tmp_path = os.path.join("uploads", safe_name)

        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(model_file.file, buffer)

        if os.path.getsize(tmp_path) > 1 * 1024 ** 3:
            os.unlink(tmp_path)
            raise HTTPException(status_code=413, detail="Model file exceeds 1 GB limit.")

        if not _validate_magic(tmp_path, ext):
            os.unlink(tmp_path)
            raise HTTPException(status_code=415, detail=f"Invalid magic bytes for {ext} file.")

        from celery_worker import run_model_scan_task
        task = run_model_scan_task.delay(tmp_path, target_class, trigger_type)
        return AsyncScanResponse(
            status="accepted",
            task_id=task.id,
            message="Model accepted for asynchronous analysis."
        )

    except HTTPException:
        raise
    except Exception:
        err_id = str(uuid.uuid4())
        logger.exception(f"[trace_id={err_id}] scan_model_async failed")
        raise HTTPException(status_code=500, detail=f"Internal error. trace_id={err_id}")


@app.get("/api/v1/scan-status/{task_id}")
@limiter.limit("60/minute")
def get_scan_status(request: Request, task_id: str, _key: str = Depends(verify_api_key)):
    from celery.result import AsyncResult
    from celery_worker import celery_app

    task_result = AsyncResult(task_id, app=celery_app)
    response = {"status": task_result.status, "task_id": task_id}

    if task_result.status == 'PENDING':
        response["message"] = "Task is waiting in queue..."
    elif task_result.status == 'PROGRESS':
        response["message"] = task_result.info.get('message', 'Processing...')
    elif task_result.status == 'SUCCESS':
        response["message"] = "Scan Complete"
        res = task_result.result
        if isinstance(res, dict):
            res['task_id'] = task_id
        response["result"] = res
    elif task_result.status == 'FAILURE':
        err_id = str(uuid.uuid4())
        logger.error(f"[trace_id={err_id}] task {task_id} failed: {task_result.info}")
        response["message"] = "Task failed to complete"
        response["error"] = f"Internal error. trace_id={err_id}"

    return response


@app.get("/api/v1/audit-report/{task_id}")
def generate_standard_audit_report(task_id: str, _key: str = Depends(verify_api_key)):
    from celery.result import AsyncResult
    from celery_worker import celery_app
    import datetime

    task_result = AsyncResult(task_id, app=celery_app)

    if not task_result.ready() or task_result.status != 'SUCCESS':
        raise HTTPException(status_code=400, detail="Report can only be generated for successful scans.")

    res = task_result.result
    details = res.get('details', {})

    report = {
        "report_metadata": {
            "version": "1.0-IARPA-JAN2026",
            "audit_timestamp": datetime.datetime.now().isoformat(),
            "task_id": task_id,
            "compliance_status": "Institutionalized AI Security Testing (IAST)"
        },
        "model_summary": {
            "architecture": details.get("architecture", "ResNet-18"),
            "framework": "ONNX Runtime" if res.get('is_onnx') else "PyTorch",
            "risk_fusion_score": res.get('fusion_risk_score'),
            "verdict": determine_risk_level(res.get('fusion_risk_score')),
            "input_shape": details.get("input_shape", "N/A"),
            "num_classes": details.get("num_classes", "N/A"),
            "parameter_count": details.get("parameter_count", "N/A")
        },
        "trojan_forensics": {
            "trigger_inversion": {
                "neural_cleanse_index": max(details.get('nc_anomaly_indices', [0.0]) or [0.0]),
                "detected_target_classes": details.get('nc_flagged_classes', [])
            },
            "test_time_checks": {
                "strip_false_acceptance": details.get('strip_fa_ratio', 0.0),
                "strip_false_rejection": details.get('strip_fr_ratio', 0.0)
            },
            "weight_analysis": {
                "max_anomaly_l2_norm": max(details.get('wa_anomaly_indices', [0.0]) or [0.0])
            },
            "natural_vulnerability_profiling": {
                "shortcut_sensitivity": details.get('natural_sensitivity', 0.0),
                "classification_drift": details.get('natural_sensitivity', 0.0)
            },
            "activation_clustering": {
                "silhouette_score": details.get('clustering_silhouette_score', 0.0)
            },
            "gradient_similarity_score": details.get('gradient_similarity', 0.0)
        },
        "strategic_recommendations": [
            "Maintain defense-in-depth across the AI supply chain.",
            "Verify model provenance for internal deployments.",
            "Conduct continuous monitoring for low-ASR backdoors."
        ]
    }

    return report


@app.get("/live")
def live():
    return {"status": "alive"}

@app.get("/ready")
def ready():
    checks = {}
    checks["api_token"] = bool(os.environ.get("API_TOKEN"))
    try:
        from celery_worker import celery_app
        celery_app.backend.client.ping()
        checks["redis"] = True
    except Exception:
        checks["redis"] = False
    all_ok = all(checks.values())
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "ready" if all_ok else "not_ready", "checks": checks}
    )

@app.get("/health")
def health_check():
    return ready()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
