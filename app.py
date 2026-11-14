
import streamlit as st
import torch
import torch.nn.functional as F
import requests

from pathlib import Path
from PIL import Image
from torchvision import transforms

from model import MHViTModel


# ================= CONFIGURATION =================

st.set_page_config(
    page_title="MH-ViT Multi-Hazard Detection",
    page_icon="🛰️",
    layout="wide"
)

MODEL_URL = (
    "https://github.com/IMK-hub/mhvit-detection-model/"
    "releases/download/detection-model/best_model.pth"
)

MODEL_PATH = Path("best_model.pth")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224

CLASS_NAMES = [
    "Flood",
    "Wildfire",
    "Landslide",
    "Normal"
]


# ================= DOWNLOAD MODEL =================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        response = requests.get(
            MODEL_URL,
            stream=True,
            timeout=(20, 300)
        )

        response.raise_for_status()

        temp_path = MODEL_PATH.with_suffix(".part")

        with open(temp_path, "wb") as file:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    file.write(chunk)

        temp_path.replace(MODEL_PATH)

    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "Invalid checkpoint format."
        )

    state_dict = checkpoint["model_state_dict"]

    num_classes = state_dict[
        "classifier.4.weight"
    ].shape[0]

    if num_classes == 4:
        classes = CLASS_NAMES
    elif num_classes == 3:
        classes = CLASS_NAMES[:3]
    else:
        raise ValueError(
            f"Unsupported number of classes: {num_classes}"
        )

    model = MHViTModel(
        num_classes=num_classes,
        use_multiscale=checkpoint.get(
            "use_multiscale", True
        ),
        use_hazard_attention=checkpoint.get(
            "use_hazard_attention", True
        ),
        use_fusion=checkpoint.get(
            "use_fusion", True
        ),
        use_confidence_head=checkpoint.get(
            "use_confidence_head",
            "confidence_head.0.weight" in state_dict
        ),
        pretrained=False
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    disaster_indices = [
        i for i, name in enumerate(classes)
        if name in [
            "Flood",
            "Wildfire",
            "Landslide"
        ]
    ]

    return model, transform, classes, disaster_indices


# ================= PREDICTION =================

def predict_image(image, model, transform,
                  classes, disaster_indices):

    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.inference_mode():

        logits, confidence = model(image_tensor)

        probabilities = F.softmax(
            logits, dim=1
        )[0]

        disaster_probs = probabilities[
            disaster_indices
        ]

        # Ignore Normal class
        disaster_probs = (
            disaster_probs /
            disaster_probs.sum()
        )

        predicted_index = torch.argmax(
            disaster_probs
        ).item()

        disaster_names = [
            classes[i] for i in disaster_indices
        ]

        predicted_class = disaster_names[
            predicted_index
        ]

        predicted_probability = (
            disaster_probs[predicted_index].item()
            * 100
        )

    return (
        predicted_class,
        predicted_probability,
        disaster_names,
        disaster_probs.cpu().tolist()
    )


# ================= USER INTERFACE =================

st.title("🛰️ MH-ViT Multi-Hazard Detection")

st.write(
    "Satellite Image-Based Multi-Hazard "
    "Prediction Using Vision Transformers"
)

st.markdown("---")

st.sidebar.title("About MH-ViT")

st.sidebar.write(
    "MH-ViT is a Vision Transformer-based "
    "satellite image classification model."
)

st.sidebar.write("**Supported Predictions:**")

st.sidebar.write("🌊 Flood")
st.sidebar.write("🔥 Wildfire")
st.sidebar.write("⛰️ Landslide")

st.sidebar.info(
    "Normal class is excluded from displayed "
    "disaster probabilities."
)

uploaded_file = st.file_uploader(
    "Upload Satellite Image",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.subheader("Prediction")

        if st.button(
            "🔍 Predict Hazard",
            type="primary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Loading MH-ViT model..."
                ):

                    model, transform, classes, indices = (
                        load_model()
                    )

                with st.spinner(
                    "Analyzing satellite image..."
                ):

                    result = predict_image(
                        image,
                        model,
                        transform,
                        classes,
                        indices
                    )

                (
                    predicted_class,
                    confidence,
                    disaster_names,
                    probabilities
                ) = result

                st.success(
                    f"Detected Hazard: {predicted_class}"
                )

                st.metric(
                    "Prediction Probability",
                    f"{confidence:.2f}%"
                )

                st.markdown(
                    "### Class-wise Probabilities"
                )

                for name, probability in zip(
                    disaster_names,
                    probabilities
                ):

                    percentage = probability * 100

                    st.write(
                        f"**{name}: {percentage:.2f}%**"
                    )

                    st.progress(
                        float(probability)
                    )

            except Exception as error:

                st.error(
                    "Prediction failed."
                )

                st.exception(error)

st.markdown("---")

st.caption(
    "MH-ViT | Multi-Hazard Satellite Image Prediction"
)
