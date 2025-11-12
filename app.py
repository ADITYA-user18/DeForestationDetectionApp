import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Deforestation Detection", page_icon="🌳", layout="wide")

st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(135deg, #224422 0%, #0f2d3a 50%, #122012 100%);
            color: #f5f7f1;
        }
        .stApp header {background: transparent;}
        div[data-testid="stFileUploader"] > label {
            font-weight: 600;
            color: #d4ead3;
        }
        .result-card {
            background: rgba(19, 40, 32, 0.7);
            border-radius: 18px;
            padding: 24px;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.25);
        }
        .image-card {
            background: rgba(20, 35, 50, 0.7);
            border-radius: 18px;
            padding: 24px;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.25);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌳 Deforestation Detection Dashboard")
st.caption(
    "Upload a satellite or aerial image to estimate the likelihood of recent deforestation. "
    "We run multiple augmented predictions for a more stable result."
)

@st.cache_resource
def load_model():
    return tf.keras.layers.TFSMLayer(
        "deforestation_model_tf",
        call_endpoint="serving_default",
    )

model = load_model()

def infer_with_tta(image: Image.Image) -> float:
    base = np.array(image, dtype=np.float32) / 255.0
    flipped = np.flip(base, axis=1)
    batch = np.stack([base, flipped], axis=0)
    outputs = model(batch)["output_0"].numpy().flatten()
    return float(outputs.mean())

with st.container():
    upload_col, info_col = st.columns([1.1, 1])

    with upload_col:
        st.subheader("Upload & Preview")
        uploaded = st.file_uploader(
            "Drag and drop or browse",
            type=["jpg", "jpeg", "png"],
            help="High-resolution imagery (≥ 224x224) improves predictions.",
        )

        if uploaded:
            img = Image.open(uploaded).convert("RGB").resize((224, 224))
            st.markdown('<div class="image-card">', unsafe_allow_html=True)
            st.image(img, caption="Normalized 224×224 preview", use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with info_col:
        st.subheader("Prediction")
        if uploaded:
            with st.spinner("Running ensemble inference..."):
                probability = infer_with_tta(img)

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.metric("Deforestation probability", f"{probability*100:.1f}%")

            st.progress(min(max(probability, 0.0), 1.0))

            if probability > 0.5:
                st.error("🔥 Likely deforestation detected in the supplied imagery.")
            else:
                st.success("🌲 Vegetation appears intact. No strong deforestation signal detected.")

            st.markdown(
                f"**Model confidence:** The ensemble average from flipping augmentations "
                f"yielded a mean score of `{probability:.3f}`."
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Upload an image to generate a prediction.")

st.markdown(
    """
    ---
    **Tips for better accuracy**
    - Use recent, cloud-free imagery with clear vegetation patterns.
    - Center the area of interest; the model currently analyses a 224×224 crop.
    - Consider multiple angles or dates to confirm trends.
    """
)
