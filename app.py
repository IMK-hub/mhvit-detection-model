# --------------------------------------------------------------
# app.py  –  Gradio web UI + model loading
# --------------------------------------------------------------
import io
import base64
import torch
import numpy as np
from PIL import Image
import gradio as gr
from model_arch import MHViTModel            # <-- your architecture file
from torchvision import transforms

# ---------- 1️⃣ Device ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 2️⃣ Model ----------
# Adjust the arguments below to match how you built the model
model = MHViTModel()                         # <- no‑args if you used defaults
model.load_state_dict(torch.load("best_mhvit.pth", map_location=DEVICE))
model.eval().to(DEVICE)

# ---------- 3️⃣ Pre‑processing (same as notebook) ----------
val_tf = transforms.Compose([
    transforms.Resize((1024, 1024)),            # original size in the paper
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]),
])

# ---------- 4️⃣ Class names ----------
CLASS_NAMES = ["Flood", "Wildfire", "Landslide", "Normal"]

# ---------- 5️⃣ Prediction function ----------
def predict(img: Image.Image):
    """
    Receives a PIL image, returns the predicted class label + confidence.
    """
    # 1️⃣ preprocess
    tensor = val_tf(img).unsqueeze(0).to(DEVICE)   # (1,3,1024,1024)

    # 2️⃣ forward
    with torch.no_grad():
        logits, _ = model(tensor)                 # _ = attention (unused)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    label    = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    # nice formatted string
    return f"**{label}** ({confidence:.1f}% confidence)", {
        name: f"{p*100:.1f}%" for name, p in zip(CLASS_NAMES, probs)
    }

# ---------- 6️⃣ Gradio Interface ----------
title = "🌍 xView‑2 Disaster Classification (MHViT)"
description = """
Upload a 1024 × 1024 RGB satellite image and the model will tell you whether it
contains **Flood**, **Wildfire**, **Landslide** or is **Normal**.  
The model was trained on the official xView‑2 challenge dataset and achieved
≈ 97 % accuracy on the external test set.
"""

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Input image (1024 × 1024)"),
    outputs=[
        gr.Label(num_top_classes=4, label="Prediction"),
        gr.JSON(label="Class probabilities")
    ],
    title=title,
    description=description,
    examples=[
        ["example_flood.png"],
        ["example_wildfire.png"],
        ["example_landslide.png"],
        ["example_normal.png"]
    ],

)

if __name__ == "__main__":
    demo.launch()