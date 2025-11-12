import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance

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
        .metric-card {
            background: rgba(30, 50, 40, 0.6);
            border-radius: 12px;
            padding: 16px;
            margin: 8px 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌳 Deforestation Detection Dashboard")
st.caption(
    "Upload a satellite or aerial image to estimate the likelihood of recent deforestation. "
    "Advanced ensemble prediction with multiple augmentations for improved accuracy."
)

@st.cache_resource
def load_model():
    return tf.keras.layers.TFSMLayer(
        "deforestation_model_tf",
        call_endpoint="serving_default",
    )

model = load_model()

def preprocess_image(image: Image.Image, size: int = 224) -> np.ndarray:
    """Enhanced preprocessing with aspect ratio preservation."""
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize maintaining aspect ratio with padding
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    
    # Create square image with padding
    new_image = Image.new('RGB', (size, size), (0, 0, 0))
    paste_x = (size - image.width) // 2
    paste_y = (size - image.height) // 2
    new_image.paste(image, (paste_x, paste_y))
    
    # Convert to numpy and normalize
    img_array = np.array(new_image, dtype=np.float32) / 255.0
    return img_array

def create_augmentations(image: Image.Image) -> list[np.ndarray]:
    """Create multiple augmented versions of the image for TTA."""
    base = preprocess_image(image)
    augmentations = [base]
    
    # Horizontal flip
    img_hflip = Image.fromarray((base * 255).astype(np.uint8))
    img_hflip = img_hflip.transpose(Image.FLIP_LEFT_RIGHT)
    augmentations.append(np.array(img_hflip, dtype=np.float32) / 255.0)
    
    # Vertical flip
    img_vflip = Image.fromarray((base * 255).astype(np.uint8))
    img_vflip = img_vflip.transpose(Image.FLIP_TOP_BOTTOM)
    augmentations.append(np.array(img_vflip, dtype=np.float32) / 255.0)
    
    # Both flips
    img_both = Image.fromarray((base * 255).astype(np.uint8))
    img_both = img_both.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    augmentations.append(np.array(img_both, dtype=np.float32) / 255.0)
    
    # Rotations (90, 180, 270)
    img_pil = Image.fromarray((base * 255).astype(np.uint8))
    for angle in [90, 180, 270]:
        img_rot = img_pil.rotate(angle, expand=False)
        augmentations.append(np.array(img_rot, dtype=np.float32) / 255.0)
    
    # Brightness adjustments
    img_pil = Image.fromarray((base * 255).astype(np.uint8))
    for factor in [0.9, 1.1]:
        enhancer = ImageEnhance.Brightness(img_pil)
        img_bright = enhancer.enhance(factor)
        augmentations.append(np.array(img_bright, dtype=np.float32) / 255.0)
    
    # Contrast adjustments
    for factor in [0.9, 1.1]:
        enhancer = ImageEnhance.Contrast(img_pil)
        img_contrast = enhancer.enhance(factor)
        augmentations.append(np.array(img_contrast, dtype=np.float32) / 255.0)
    
    # Saturation adjustments (for color variation)
    for factor in [0.9, 1.1]:
        enhancer = ImageEnhance.Color(img_pil)
        img_sat = enhancer.enhance(factor)
        augmentations.append(np.array(img_sat, dtype=np.float32) / 255.0)
    
    return augmentations

def infer_with_advanced_tta(image: Image.Image) -> tuple[float, dict]:
    """Advanced TTA with multiple augmentations and ensemble prediction."""
    # Create all augmentations
    augmentations = create_augmentations(image)
    
    # Batch predictions (process in batches to avoid memory issues)
    batch_size = 8
    all_predictions = []
    
    for i in range(0, len(augmentations), batch_size):
        batch = augmentations[i:i+batch_size]
        batch_array = np.stack(batch, axis=0)
        outputs = model(batch_array)["output_0"].numpy().flatten()
        all_predictions.extend(outputs.tolist())
    
    # Calculate statistics
    mean_pred = float(np.mean(all_predictions))
    median_pred = float(np.median(all_predictions))
    std_pred = float(np.std(all_predictions))
    min_pred = float(np.min(all_predictions))
    max_pred = float(np.max(all_predictions))
    
    # Use weighted average (original image gets higher weight)
    weights = [2.0] + [1.0] * (len(all_predictions) - 1)  # Original gets 2x weight
    weighted_mean = float(np.average(all_predictions, weights=weights))
    
    # Confidence score (inverse of std, normalized)
    confidence = max(0.0, min(1.0, 1.0 - std_pred * 2))
    
    stats = {
        'mean': mean_pred,
        'median': median_pred,
        'weighted_mean': weighted_mean,
        'std': std_pred,
        'min': min_pred,
        'max': max_pred,
        'confidence': confidence,
        'num_augmentations': len(all_predictions),
        'all_predictions': all_predictions
    }
    
    # Use weighted mean as primary prediction
    return weighted_mean, stats

with st.container():
    upload_col, info_col = st.columns([1.1, 1])

    with upload_col:
        st.subheader("📤 Upload & Preview")
        uploaded = st.file_uploader(
            "Drag and drop or browse",
            type=["jpg", "jpeg", "png"],
            help="High-resolution imagery (≥ 224x224) improves predictions. Multiple augmentations are applied for enhanced accuracy.",
        )

        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            # Show original image
            st.markdown('<div class="image-card">', unsafe_allow_html=True)
            st.image(img, caption="Original image", use_column_width=True)
            
            # Show preprocessed version
            preprocessed = preprocess_image(img)
            preprocessed_display = Image.fromarray((preprocessed * 255).astype(np.uint8))
            st.image(preprocessed_display, caption="Preprocessed 224×224 (with padding)", use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with info_col:
        st.subheader("🎯 Prediction Results")
        if uploaded:
            with st.spinner("Running advanced ensemble inference with multiple augmentations..."):
                probability, stats = infer_with_advanced_tta(img)

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            
            # Main prediction
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Deforestation Probability", f"{probability*100:.1f}%", 
                         delta=f"±{stats['std']*100:.1f}%", delta_color="off")
            with col2:
                st.metric("Confidence", f"{stats['confidence']*100:.1f}%")
            
            st.progress(min(max(probability, 0.0), 1.0))

            # Prediction result
            if probability > 0.6:
                st.error(f"🔥 **High likelihood of deforestation detected** (score: {probability:.3f})")
            elif probability > 0.4:
                st.warning(f"⚠️ **Moderate deforestation signal** (score: {probability:.3f})")
            else:
                st.success(f"🌲 **Vegetation appears intact** (score: {probability:.3f})")
            
            st.markdown("---")
            
            # Detailed statistics
            with st.expander("📊 Detailed Statistics", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean", f"{stats['mean']:.3f}")
                    st.metric("Median", f"{stats['median']:.3f}")
                with col2:
                    st.metric("Min", f"{stats['min']:.3f}")
                    st.metric("Max", f"{stats['max']:.3f}")
                with col3:
                    st.metric("Std Dev", f"{stats['std']:.3f}")
                    st.metric("Augmentations", f"{stats['num_augmentations']}")
            
            # Prediction distribution
            st.markdown("**Prediction Distribution Across Augmentations:**")
            # Create visualization as dictionary for bar chart
            pred_dict = {f'A{i+1}': pred for i, pred in enumerate(stats['all_predictions'])}
            st.line_chart(pred_dict, height=200)
            st.caption(f"Range: {stats['min']:.3f} - {stats['max']:.3f} | Mean: {stats['mean']:.3f} | Std: {stats['std']:.3f}")
            
            st.markdown(
                f"""
                **Ensemble Method:** Weighted average from {stats['num_augmentations']} augmented predictions
                - Original image: 2x weight
                - Augmentations: Horizontal/vertical flips, rotations (90°/180°/270°), brightness/contrast/saturation adjustments
                - Confidence: {stats['confidence']*100:.1f}% (based on prediction consistency across augmentations)
                """
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("👆 Upload an image to generate a prediction with enhanced accuracy.")

st.markdown(
    """
    ---
    ### 🎯 Accuracy Enhancement Features
    
    **Advanced Test-Time Augmentation (TTA):**
    - ✅ Multiple augmentations (flips, rotations, brightness, contrast, saturation)
    - ✅ Weighted ensemble averaging (original image weighted 2x for stability)
    - ✅ Confidence scoring based on prediction consistency across augmentations
    - ✅ Enhanced preprocessing with aspect ratio preservation and proper padding
    - ✅ Batch processing for efficient inference
    
    **Tips for better accuracy:**
    - Use recent, cloud-free imagery with clear vegetation patterns
    - Ensure high-resolution images (≥ 224x224 pixels)
    - Center the area of interest for best results
    - Consider multiple angles or dates to confirm trends
    - The model uses ensemble predictions across multiple augmentations for improved reliability
    """
)
