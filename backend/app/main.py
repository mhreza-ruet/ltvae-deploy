from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from .types import PredictRequest, PredictResponse, HealthResponse
from .model import ModelManager

# Debug + utils
import os, base64, io, logging, traceback
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ltvae")

app = FastAPI(
    title="LTVAE: Latent Generator and Properties Predictor from SMILES",
    version="0.8.1"
)
model_mgr = ModelManager()

@app.get("/health", response_model=HealthResponse, tags=["Default"], summary="")
def health():
    try:
        model_mgr._ensure_loaded()
    except Exception:
        pass
    return HealthResponse(status="ok", model_loaded=model_mgr.is_loaded())

def _compose_preview_png_only_smiles(b64_png: str, smiles: str) -> bytes:
    """Return a PNG that just shows the RDKit molecule image (no extra caption)."""
    mol_img = Image.open(io.BytesIO(base64.b64decode(b64_png))).convert("RGB")
    out = io.BytesIO()
    mol_img.save(out, format="PNG")
    return out.getvalue()

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Default"],
    summary="Predicts the Latent Vector and Properties or Returns a 2D Molecule Graph When View=True"
)
def predict(
    req: PredictRequest,
    view: bool = Query(False, description="If true, returns a 2D molecule graph")
):
    try:
        logger.info("Predict called with SMILES: %s | view=%s", req.smiles, view)
        result = model_mgr.predict(req.smiles)
        logger.info("Predict result meta: %s", result.get("meta"))

        # PNG view (no properties on the image) when view=true
        if view:
            if not result["valid"] or not result.get("image_png_b64"):
                raise HTTPException(status_code=400, detail="Invalid SMILES or image unavailable")
            png_bytes = _compose_preview_png_only_smiles(
                b64_png=result["image_png_b64"], smiles=req.smiles
            )
            return Response(content=png_bytes, media_type="image/png")

        # Default JSON (no base64 in schema)
        return PredictResponse(
            input=req.smiles,
            valid=result["valid"],
            latent=result["latent"],
            properties=result["properties"],
            meta=result["meta"],
        )

    except ValueError as e:
        logger.exception("Validation error in /predict: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Unhandled error in /predict: %s\n%s", e, tb)
        # Surface the error & traceback so we can see the exact root cause
        raise HTTPException(status_code=500, detail=f"Internal error: {e}\n{tb}")

@app.get("/")
def root():
    return {"service": app.title, "version": app.version}

# --- Debug endpoint to inspect env + model load status ---
@app.get("/debug/status", tags=["Default"], summary="")
def debug_status():
    return {
        "service": app.title,
        "version": app.version,
        "env": {
            "MODEL_PATH": os.getenv("MODEL_PATH"),
            "VOCAB_JSON": os.getenv("VOCAB_JSON"),
            "PROP_HEAD_PATH": os.getenv("PROP_HEAD_PATH"),
        },
        "model": {
            "loaded": model_mgr.is_loaded(),
            "is_real": getattr(model_mgr, "_is_real", None),
            "has_prop": getattr(model_mgr, "_has_prop", None),
            "last_error": getattr(model_mgr, "_last_error", None),
        }
    }

# ---- Clean up the docs (hide 422/500 blocks) ----
from fastapi.openapi.utils import get_openapi
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
        description="LTVAE: Latent + Properties. Use ?view=true on POST /predict for a PNG with the 2D image.",
    )
    for path_item in openapi_schema.get("paths", {}).values():
        for op in path_item.values():
            if isinstance(op, dict):
                op.get("responses", {}).pop("422", None)
                op.get("responses", {}).pop("500", None)
    app.openapi_schema = openapi_schema
    return openapi_schema

app.openapi = custom_openapi
