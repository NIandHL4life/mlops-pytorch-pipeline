"""
FastAPI Model Serving Application.
Exposes:
  - GET /health: Health probe endpoint for Kubernetes liveness/readiness
  - POST /predict: Inference endpoint accepting image files and returning class probabilities
  - GET /info: Metadata about loaded model architecture and configuration
"""

import io
import os
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn.functional as F

from dataset import CIFAR10_CLASSES, get_transforms
from model import get_model

app = FastAPI(
    title="PyTorch CIFAR-10 Serving API",
    description="Production-grade model serving endpoint with health checks and probability distributions",
    version="1.0.0",
)

# Global model state
MODEL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_METADATA: Dict[str, Any] = {}


def load_model_checkpoint() -> bool:
    """
    Attempts to locate and load the saved PyTorch checkpoint.
    Looks in standard checkpoint paths.
    """
    global MODEL, MODEL_METADATA

    checkpoint_candidates = [
        Path(os.getenv("CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt")),
        Path("checkpoints/classifier_v1.pt"),
        Path("/app/checkpoints/best_model.pt"),
    ]

    checkpoint_path = None
    for candidate in checkpoint_candidates:
        if candidate.exists() and candidate.is_file():
            checkpoint_path = candidate
            break

    if checkpoint_path is None:
        # If no checkpoint exists yet, initialize default architecture for graceful readiness
        print("No checkpoint found on disk. Initializing base model weights.")
        MODEL = get_model(architecture="resnet18", num_classes=10).to(DEVICE)
        MODEL.eval()
        MODEL_METADATA = {
            "status": "uninitialized_weights",
            "architecture": "resnet18",
            "checkpoint_path": None,
        }
        return True

    try:
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(str(checkpoint_path), map_location=DEVICE)
        
        arch = checkpoint.get("architecture", "resnet18")
        num_classes = checkpoint.get("num_classes", 10)
        
        model = get_model(architecture=arch, num_classes=num_classes).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        MODEL = model
        MODEL_METADATA = {
            "status": "loaded",
            "architecture": arch,
            "num_classes": num_classes,
            "val_loss": checkpoint.get("val_loss"),
            "val_accuracy": checkpoint.get("val_accuracy"),
            "epoch": checkpoint.get("epoch"),
            "checkpoint_path": str(checkpoint_path),
        }
        print(f"Model successfully loaded. Best val accuracy: {checkpoint.get('val_accuracy')}")
        return True
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False


@app.on_event("startup")
def startup_event():
    """Load model weights on application startup."""
    load_model_checkpoint()


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """
    Kubernetes Liveness and Readiness probe endpoint.
    Returns HTTP 200 if the service is operational and model is initialized.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded yet.",
        )
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(DEVICE),
        "metadata": MODEL_METADATA,
    }


@app.get("/info")
def model_info():
    """Returns metadata about the currently served model."""
    return {
        "classes": CIFAR10_CLASSES,
        "input_resolution": "32x32",
        "device": str(DEVICE),
        "metadata": MODEL_METADATA,
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Accepts an input image, applies CIFAR-10 evaluation transforms,
    and returns sorted class probabilities and top prediction.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready for inference.",
        )

    # Validate content type
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a valid image.",
        )

    try:
        # Read and decode image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Resize to 32x32 if needed
        if pil_image.size != (32, 32):
            pil_image = pil_image.resize((32, 32), Image.Resampling.BILINEAR)

        # Apply inference transforms
        transform = get_transforms(train=False)
        input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

        # Run inference
        with torch.no_grad():
            logits = MODEL(input_tensor)
            probabilities = F.softmax(logits, dim=1).squeeze(0)

        # Map to class probabilities
        probs_dict = {
            class_name: round(float(prob), 4)
            for class_name, prob in zip(CIFAR10_CLASSES, probabilities)
        }

        # Top prediction
        top_prob, top_idx = torch.max(probabilities, dim=0)
        top_class = CIFAR10_CLASSES[top_idx.item()]

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "prediction": top_class,
                "confidence": round(float(top_prob), 4),
                "probabilities": probs_dict,
                "filename": image.filename,
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )
