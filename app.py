
import streamlit as st
import torch
import torch.nn.functional as F
import requests
import html

from pathlib import Path
from PIL import Image
from torchvision import transforms

from model import MHViTModel


# =====================================================
# 1. CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="MH-ViT Detection Model | DisasterVision AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
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

CLASS_INFO = {
    "Flood": {
        "icon": "🌊",
        "color": "#22d3ee"
    },
    "Wildfire": {
        "icon": "🔥",
        "color": "#60a5fa"
    },
    "Landslide": {
        "icon": "⛰️",
        "color": "#818cf8"
    }
}


# =====================================================
# 2. CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* Hide Streamlit Header */
header[data-testid="stHeader"] {
    display: none !important;
}

/* Hide Streamlit Toolbar */
[data-testid="stToolbar"] {
    display: none !important;
}

/* Hide Deploy Button */
[data-testid="stAppDeployButton"] {
    display: none !important;
}

/* Hide Main Menu */
#MainMenu {
    visibility: hidden;
}

/* Hide Footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #071936;
    --blue: #1559a9;
    --cyan: #22d3ee;
    --light: #edf6ff;
    --text: #123568;
}

header[data-testid="stHeader"] {
    display: none !important;
}

[data-testid="stToolbar"] {
    display: none !important;
}

[data-testid="stAppDeployButton"] {
    display: none !important;
}

#MainMenu,
footer {
    visibility: hidden;
}

.stApp {
    background: linear-gradient(
        135deg,
        #edf5ff,
        #ffffff,
        #eaf3ff
    );
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

.block-container {
    max-width: 1500px;
    padding-top: 0.7rem;
    padding-bottom: 0.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

div[data-testid="stVerticalBlock"] {
    gap: 0.55rem;
}

/* HERO HEADER */

.hero-container {
    background: linear-gradient(
        110deg,
        #071936,
        #103b7a,
        #1559a9
    );
    padding: 18px 24px;
    border-radius: 13px;
    border: 1px solid #285c9d;
    box-shadow: 0 5px 18px rgba(15,65,130,0.15);
    margin-bottom: 20px;
}

.hero-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
}

.hero-title {
    font-size: 25px;
    font-weight: 800;
    color: white;
    margin: 0;
    letter-spacing: 0.3px;
}

.hero-subtitle {
    color: #c3dcff;
    font-size: 11px;
    margin-top: 5px;
}

.header-status {
    background: rgba(34,211,238,0.12);
    border: 1px solid #22d3ee;
    color: #67e8f9;
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 11px;
    font-weight: 700;
}

/* SECTION HEADINGS */

.section-heading {
    color: #103568;
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 3px;
}

.section-subtitle {
    color: #8496ad;
    font-size: 11px;
    margin-bottom: 12px;
}

/* LEFT PANEL */

.left-panel {
    background: white;
    border: 1px solid #d6e6f8;
    border-radius: 13px;
    padding: 15px;
    box-shadow: 0 4px 15px rgba(15,55,110,0.06);
    min-height: 470px;
}

/* RIGHT ASSESSMENT PANEL */

.assessment-card {
    background: linear-gradient(
        145deg,
        #071c42,
        #10346e
    );
    border: 1px solid #2861a5;
    border-radius: 13px;
    padding: 18px;
    color: white;
    box-shadow: 0 5px 18px rgba(10,45,100,0.15);
}

.assessment-title {
    font-size: 17px;
    font-weight: 700;
    color: white;
    margin-bottom: 14px;
}

.threat-box {
    background: linear-gradient(
        120deg,
        #123b76,
        #174b8c
    );
    border: 1px solid #3976ba;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 16px;
}

.threat-label {
    font-size: 22px;
    font-weight: 800;
    margin-top: 7px;
}

.confidence {
    font-size: 29px;
    font-weight: 800;
    color: white;
}

.confidence-label {
    color: #c3dcff;
    font-size: 11px;
    margin-left: 5px;
}

/* PROBABILITY BARS */

.prob-heading {
    font-size: 13px;
    font-weight: 700;
    color: white;
    margin-bottom: 13px;
}

.prob-row {
    margin-bottom: 15px;
}

.prob-title {
    display: flex;
    justify-content: space-between;
    color: white;
    font-size: 11px;
    margin-bottom: 6px;
}

.prob-track {
    width: 100%;
    height: 7px;
    background: #294c7b;
    border-radius: 20px;
    overflow: hidden;
}

.prob-fill {
    height: 100%;
    border-radius: 20px;
}

/* EMPTY RESULT */

.empty-title {
    font-size: 21px;
    font-weight: 800;
    color: #7dd3fc;
    margin-top: 10px;
}

.empty-description {
    color: #b9d9ff;
    font-size: 11px;
    margin-top: 8px;
    line-height: 1.6;
}

/* BOTTOM INFORMATION */

.info-strip {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    background: white;
    border: 1px solid #d6e6f8;
    border-radius: 11px;
    padding: 13px 17px;
    margin-top: 16px;
    box-shadow: 0 3px 12px rgba(15,55,110,0.05);
}

.info-item {
    color: #7188a5;
    font-size: 10px;
}

.info-item b {
    color: #123c78;
    font-size: 12px;
    display: block;
    margin-top: 4px;
}

/* FILE UPLOADER */

[data-testid="stFileUploader"] {
    background: #f5faff;
    border: 1px dashed #6da4dc;
    border-radius: 10px;
    padding: 8px;
    margin-bottom: 8px;
}

/* IMAGE */

[data-testid="stImage"] img {
    max-height: 390px;
    width: 100%;
    object-fit: contain;
    border-radius: 10px;
    border: 1px solid #d3e4f6;
    background: #f0f6fc;
}

/* PREDICTION BUTTON */

.stButton > button {
    background: linear-gradient(
        110deg,
        #103b7a,
        #1559a9
    );
    color: white;
    border: 1px solid #285c9d;
    border-radius: 8px;
    font-weight: 700;
    padding: 10px;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #071936;
    color: white;
    border-color: #22d3ee;
}

/* RESPONSIVE */

@media (max-width: 768px) {

    .block-container {
        padding-left: 0.8rem;
        padding-right: 0.8rem;
    }

    .hero-title {
        font-size: 20px;
    }

    .left-panel {
        min-height: auto;
    }

    [data-testid="stImage"] img {
        max-height: 320px;
    }

}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. MODEL LOADING
# =====================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        with st.spinner("Downloading MH-ViT model..."):

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

    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint format.")

    if "model_state_dict" not in checkpoint:
        raise ValueError("Invalid checkpoint format.")

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

    return (
        model,
        transform,
        classes,
        disaster_indices
    )


# =====================================================
# 4. PREDICTION FUNCTION
# =====================================================

def predict_image(
    image,
    model,
    transform,
    classes,
    disaster_indices
):

    image_tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():

        logits, confidence = model(image_tensor)

        probabilities = F.softmax(
            logits,
            dim=1
        )[0]

        # Ignore Normal class

        disaster_probs = probabilities[
            disaster_indices
        ]

        disaster_probs = (
            disaster_probs /
            disaster_probs.sum().clamp_min(1e-12)
        )

        predicted_index = torch.argmax(
            disaster_probs
        ).item()

        disaster_names = [
            classes[i]
            for i in disaster_indices
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


# =====================================================
# 5. HEADER
# =====================================================

st.markdown("""
<div class="hero-container">

    <div class="hero-content">

        <div>

            <div class="hero-title">
                🛰️ MHViT Detection Model
            </div>

            <div class="hero-subtitle">
                DisasterVision AI | Satellite Image-Based
                Natural Disaster Classification System
            </div>

        </div>

        <div class="header-status">
            ● MODEL ONLINE
        </div>

    </div>

</div>
""", unsafe_allow_html=True)


# =====================================================
# 6. MAIN DASHBOARD
# =====================================================

col1, col2 = st.columns(
    [1.15, 0.85],
    gap="medium"
)


# =====================================================
# 7. LEFT PANEL - IMAGE UPLOAD
# =====================================================

with col1:

    st.markdown("""
    <div style="margin-bottom:9px;">

        <div class="section-heading">
            🛰️ Satellite Telemetry Viewport
        </div>

        <div class="section-subtitle">
            Satellite image input and visualization
        </div>

    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Satellite Image (PNG / JPG / TIFF)",
        type=["png", "jpg", "jpeg", "tif", "tiff"]
    )

    image = None

    if uploaded_file is not None:

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.image(
                image,
                caption="Satellite Image - Target Region",
                use_container_width=True
            )

        except Exception as error:

            st.error(
                f"Unable to read image: {error}"
            )

    else:

        st.markdown("""
        <div style="
            background:white;
            border:1px dashed #78a9dc;
            border-radius:12px;
            padding:70px 15px;
            text-align:center;
            margin-top:5px;">

            <div style="font-size:48px;">
                🛰️
            </div>

            <h3 style="
                color:#174477;
                font-size:18px;">
                Satellite Image Viewport
            </h3>

            <p style="
                color:#8496ad;
                font-size:12px;">
                Upload a satellite image to start
                disaster classification.
            </p>

            <p style="
                color:#9aabc0;
                font-size:10px;">
                Supported: PNG | JPG | JPEG | TIFF
            </p>

        </div>
        """, unsafe_allow_html=True)


# =====================================================
# 8. RIGHT PANEL - AI THREAT ASSESSMENT
# =====================================================

with col2:

    st.markdown("""
    <div class="section-heading"
         style="margin-bottom:12px;">
        ⚡ AI Threat Assessment
    </div>
    """, unsafe_allow_html=True)

    if image is not None:

        predict_clicked = st.button(
            "🔍 Predict Hazard",
            type="primary",
            use_container_width=True
        )

        if predict_clicked:

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

                info = CLASS_INFO[predicted_class]

                # Build probability bars

                probability_html = ""

                for name, probability in zip(
                    disaster_names,
                    probabilities
                ):

                    percentage = probability * 100

                    class_info = CLASS_INFO[name]

                    probability_html += f"""
                    <div class="prob-row">

                        <div class="prob-title">

                            <span>
                                {class_info['icon']}
                                {html.escape(name)}
                            </span>

                            <b>{percentage:.2f}%</b>

                        </div>

                        <div class="prob-track">

                            <div class="prob-fill"
                                 style="
                                 width:{percentage:.2f}%;
                                 background:{class_info['color']};">
                            </div>

                        </div>

                    </div>
                    """

                st.markdown(f"""
                <div class="assessment-card">

                    <div class="assessment-title">
                        AI Classification Result
                    </div>

                    <div class="threat-box">

                        <div style="
                            color:#b9d9ff;
                            font-size:11px;
                            font-weight:600;">

                            DETECTED DISASTER CATEGORY

                        </div>

                        <div class="threat-label"
                             style="color:{info['color']};">

                            {info['icon']}
                            {html.escape(predicted_class.upper())}

                        </div>

                        <div style="margin-top:10px;">

                            <span class="confidence">
                                {confidence:.2f}%
                            </span>

                            <span class="confidence-label">
                                Prediction Probability
                            </span>

                        </div>

                    </div>

                    <div class="prob-heading">
                        Category Distribution
                    </div>

                    {probability_html}

                </div>
                """, unsafe_allow_html=True)

            except Exception as error:

                st.error("Prediction failed.")

                st.exception(error)

        else:

            st.markdown("""
            <div class="assessment-card">

                <div class="assessment-title">
                    AI Classification Result
                </div>

                <div class="threat-box">

                    <div style="
                        color:#b9d9ff;
                        font-size:11px;">

                        DETECTION STATUS

                    </div>

                    <div class="empty-title">
                        AWAITING ANALYSIS
                    </div>

                    <div class="empty-description">
                        Click Predict Hazard to initiate
                        MH-ViT satellite image analysis.
                    </div>

                </div>

                <div class="prob-heading">
                    Supported Disaster Categories
                </div>

                <div style="
                    color:#7dd3fc;
                    font-size:12px;
                    margin:12px 0;">
                    🌊 Flood Detection
                </div>

                <div style="
                    color:#93c5fd;
                    font-size:12px;
                    margin:12px 0;">
                    🔥 Wildfire Detection
                </div>

                <div style="
                    color:#a5b4fc;
                    font-size:12px;
                    margin:12px 0;">
                    ⛰️ Landslide Detection
                </div>

            </div>
            """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="assessment-card">

            <div class="assessment-title">
                AI Classification Result
            </div>

            <div class="threat-box">

                <div style="
                    color:#b9d9ff;
                    font-size:11px;">

                    DETECTION STATUS

                </div>

                <div class="empty-title">
                    AWAITING IMAGE
                </div>

                <div class="empty-description">
                    Upload satellite imagery to initiate
                    MH-ViT analysis.
                </div>

            </div>

            <div class="prob-heading">
                Supported Disaster Categories
            </div>

            <div style="
                color:#7dd3fc;
                font-size:12px;
                margin:12px 0;">
                🌊 Flood Detection
            </div>

            <div style="
                color:#93c5fd;
                font-size:12px;
                margin:12px 0;">
                🔥 Wildfire Detection
            </div>

            <div style="
                color:#a5b4fc;
                font-size:12px;
                margin:12px 0;">
                ⛰️ Landslide Detection
            </div>

        </div>
        """, unsafe_allow_html=True)


# =====================================================
# 9. BOTTOM SYSTEM INFORMATION
# =====================================================

st.markdown(f"""
<div class="info-strip">

    <div class="info-item">
        MODEL ARCHITECTURE
        <b>MH-ViT</b>
    </div>

    <div class="info-item">
        INPUT RESOLUTION
        <b>224 × 224</b>
    </div>

    <div class="info-item">
        IMAGE SENSOR
        <b>RGB</b>
    </div>

    <div class="info-item">
        CLASSIFICATION
        <b>3 Disaster Classes</b>
    </div>

    <div class="info-item">
        COMPUTE DEVICE
        <b>{str(DEVICE).upper()}</b>
    </div>

</div>
""", unsafe_allow_html=True)


# =====================================================
# 10. FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "MH-ViT | Multi-Hazard Satellite Image Prediction"
)
