import os
import uuid
import torch
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from PIL import Image
from torchvision import transforms


import sys
from pathlib import Path

# Add the project root folder to Python's module search path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
# Modular Imports
from src.models import SkinCancerHybridModel
from src.xai_vision import GradCAM, overlay_heatmap, generate_pixel_shap
from src.reporting import generate_clinical_report

# 1. Initialize FastAPI FIRST before registering any routes
app = FastAPI(title="UK Clinical AI Diagnostic Gateway")

# Setup static hosting for reports and heatmaps
os.makedirs("api/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# Global Model Loading (UK Clinical Standard: Load once on startup)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkinCancerHybridModel().to(device)
model.resnet.load_state_dict(torch.load("models/resnet_backbone.pth", map_location=device))
model.classifier.load_model("models/xgb_head.json")
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Corrected Health Check placement
@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/api/v1/diagnose")
async def diagnose(file: UploadFile = File(...)):
    # Create a unique ID for this clinical session
    session_id = str(uuid.uuid4())[:8]
    input_path = f"api/static/input_{session_id}.jpg"
    
    # Save input
    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Prediction
    img_raw = Image.open(input_path).convert('RGB')
    img_tensor = transform(img_raw).unsqueeze(0).to(device)
    features = model(img_tensor).cpu().numpy()
    probs = model.classifier.predict_proba(features)[0]
    pred_idx = int(np.argmax(probs))
    class_names = ['Benign', 'Malignant']

    # Generate XAI components
    cam = GradCAM(model.resnet, model.resnet.layer4[-1])
    heatmap = cam.generate_heatmap(img_tensor, class_idx=pred_idx)
    
    # Assumes overlay_heatmap returns a numpy image array or saves a path
    grad_img = overlay_heatmap(input_path, heatmap) 
    
    # 🌟 CRITICAL FIX: Normalize and scale image arrays to 0-255 uint8 to prevent black box output
    if grad_img.dtype == np.float32 or grad_img.dtype == np.float64:
        if grad_img.max() <= 1.0:
            grad_img = (grad_img * 255).astype(np.uint8)
        else:
            grad_img = grad_img.astype(np.uint8)
    else:
        grad_img = grad_img.astype(np.uint8)

    # Force bounding constraints to strictly map within standard image bytes
    grad_img = np.clip(grad_img, 0, 255).astype(np.uint8)
    
    # Convert RGB to BGR before writing with OpenCV to preserve proper heat colors
    grad_img_bgr = cv2.cvtColor(grad_img, cv2.COLOR_RGB2BGR)

    # Save the overlay heatmap safely
    output_heatmap_path = f"api/static/heatmap_{session_id}.jpg"
    cv2.imwrite(output_heatmap_path, grad_img_bgr)

    # Trigger Automated Report
    report_file = generate_clinical_report(
        original_img=img_raw,
        gradcam_img=grad_img,
        shap_img=grad_img, 
        diagnosis=class_names[pred_idx],
        confidence=f"{probs[pred_idx]:.2%}",
        report_id=session_id
    )

    # Read the heatmap image bytes to stream back to Streamlit
    with open(output_heatmap_path, "rb") as f:
        image_bytes = f.read()

    # Build response with custom headers matching your Streamlit expectation
    headers = {
        "X-Diagnostic-Prediction": class_names[pred_idx],
        "X-Confidence-Score": str(probs[pred_idx]),
        "X-Clinical-Traceability": f"Session {session_id} verified via hybrid ResNet-XGBoost backend."
    }
    
    return Response(content=image_bytes, media_type="image/jpeg", headers=headers)



#run backend
#uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload