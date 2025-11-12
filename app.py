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

def preprocess_image(image: Image.Image, size: int = 224, use_imagenet: bool = True) -> np.ndarray:
    """Enhanced preprocessing with ImageNet normalization for proper model input."""
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize maintaining aspect ratio with padding
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    
    # Create square image with padding (use black padding)
    new_image = Image.new('RGB', (size, size), (0, 0, 0))
    paste_x = (size - image.width) // 2
    paste_y = (size - image.height) // 2
    new_image.paste(image, (paste_x, paste_y))
    
    # Convert to numpy and normalize to 0-1 range
    img_array = np.array(new_image, dtype=np.float32) / 255.0
    
    # Apply ImageNet normalization (mean subtraction and std division)
    if use_imagenet:
        # ImageNet mean and std
        imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # Normalize: (x - mean) / std
        img_array = (img_array - imagenet_mean) / imagenet_std
    
    return img_array

def create_augmentations(image: Image.Image, use_imagenet: bool = True) -> list[np.ndarray]:
    """Create multiple augmented versions of the image for TTA with proper preprocessing."""
    # First, resize and pad the image (before normalization)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    size = 224
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    new_image = Image.new('RGB', (size, size), (0, 0, 0))
    paste_x = (size - image.width) // 2
    paste_y = (size - image.height) // 2
    new_image.paste(image, (paste_x, paste_y))
    
    # ImageNet normalization constants
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # Helper function to normalize
    def normalize_img(img_arr):
        if use_imagenet:
            return (img_arr - imagenet_mean) / imagenet_std
        return img_arr
    
    # Base image (0-1 range)
    base_01 = np.array(new_image, dtype=np.float32) / 255.0
    base = normalize_img(base_01)
    augmentations = [base]
    
    # Horizontal flip
    img_hflip = new_image.transpose(Image.FLIP_LEFT_RIGHT)
    hflip_01 = np.array(img_hflip, dtype=np.float32) / 255.0
    augmentations.append(normalize_img(hflip_01))
    
    # Vertical flip
    img_vflip = new_image.transpose(Image.FLIP_TOP_BOTTOM)
    vflip_01 = np.array(img_vflip, dtype=np.float32) / 255.0
    augmentations.append(normalize_img(vflip_01))
    
    # Both flips
    img_both = new_image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    both_01 = np.array(img_both, dtype=np.float32) / 255.0
    augmentations.append(normalize_img(both_01))
    
    # Rotations (90, 180, 270)
    for angle in [90, 180, 270]:
        img_rot = new_image.rotate(angle, expand=False)
        rot_01 = np.array(img_rot, dtype=np.float32) / 255.0
        augmentations.append(normalize_img(rot_01))
    
    # Brightness adjustments
    for factor in [0.9, 1.1]:
        enhancer = ImageEnhance.Brightness(new_image)
        img_bright = enhancer.enhance(factor)
        bright_01 = np.array(img_bright, dtype=np.float32) / 255.0
        augmentations.append(normalize_img(bright_01))
    
    # Contrast adjustments
    for factor in [0.9, 1.1]:
        enhancer = ImageEnhance.Contrast(new_image)
        img_contrast = enhancer.enhance(factor)
        contrast_01 = np.array(img_contrast, dtype=np.float32) / 255.0
        augmentations.append(normalize_img(contrast_01))
    
    # Saturation adjustments (for color variation)
    for factor in [0.9, 1.1]:
        enhancer = ImageEnhance.Color(new_image)
        img_sat = enhancer.enhance(factor)
        sat_01 = np.array(img_sat, dtype=np.float32) / 255.0
        augmentations.append(normalize_img(sat_01))
    
    return augmentations

def infer_with_advanced_tta(image: Image.Image, invert_output: bool = False) -> tuple[float, dict]:
    """Advanced TTA with multiple augmentations and ensemble prediction.
    
    Args:
        image: Input image to process
        invert_output: If True, invert the output (1 - prediction). 
                      Use this if model outputs vegetation probability instead of deforestation.
    """
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
    
    # Invert if needed (if model outputs vegetation probability)
    if invert_output:
        all_predictions = [1.0 - p for p in all_predictions]
    
    # Calculate statistics
    mean_pred = float(np.mean(all_predictions))
    median_pred = float(np.median(all_predictions))
    std_pred = float(np.std(all_predictions))
    min_pred = float(np.min(all_predictions))
    max_pred = float(np.max(all_predictions))
    
    # Use weighted average (original image gets higher weight)
    weights = [2.0] + [1.0] * (len(all_predictions) - 1)  # Original gets 2x weight
    weighted_mean = float(np.average(all_predictions, weights=weights))
    
    # Confidence score (inverse of std, normalized) - higher std = lower confidence
    # Normalize std to 0-1 range (assuming std is typically 0-0.5)
    confidence = max(0.0, min(1.0, 1.0 - (std_pred / 0.25)))
    
    stats = {
        'mean': mean_pred,
        'median': median_pred,
        'weighted_mean': weighted_mean,
        'std': std_pred,
        'min': min_pred,
        'max': max_pred,
        'confidence': confidence,
        'num_augmentations': len(all_predictions),
        'all_predictions': all_predictions,
        'range': max_pred - min_pred  # Add range for debugging
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
            
            # Show preprocessed version (for display, show before ImageNet normalization)
            size = 224
            display_img = img.copy()
            display_img.thumbnail((size, size), Image.Resampling.LANCZOS)
            display_new = Image.new('RGB', (size, size), (0, 0, 0))
            paste_x = (size - display_img.width) // 2
            paste_y = (size - display_img.height) // 2
            display_new.paste(display_img, (paste_x, paste_y))
            st.image(display_new, caption="Preprocessed 224×224 (with padding, before ImageNet normalization)", use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with info_col:
        st.subheader("🎯 Prediction Results")
        if uploaded:
            with st.spinner("Running advanced ensemble inference with multiple augmentations..."):
                # Try with inverted output first (model might output vegetation probability)
                # If model outputs vegetation prob, then deforestation = 1 - vegetation
                probability, stats = infer_with_advanced_tta(img, invert_output=True)
                
                # Check if variance is too low - if so, the model might not be working correctly
                if stats['std'] < 0.005:  # Very low variance threshold
                    st.error(
                        f"⚠️ **Model Output Issue Detected:** "
                        f"Prediction variance is extremely low (std: {stats['std']:.4f}). "
                        f"The model is producing nearly identical predictions for all inputs, "
                        f"which suggests it may not be properly trained or needs different preprocessing."
                    )

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
            with st.expander("📊 Detailed Statistics", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean", f"{stats['mean']:.4f}")
                    st.metric("Median", f"{stats['median']:.4f}")
                    st.metric("Range", f"{stats.get('range', stats['max'] - stats['min']):.4f}")
                with col2:
                    st.metric("Min", f"{stats['min']:.4f}")
                    st.metric("Max", f"{stats['max']:.4f}")
                    st.metric("Std Dev", f"{stats['std']:.4f}")
                with col3:
                    st.metric("Augmentations", f"{stats['num_augmentations']}")
                    st.metric("Weighted Mean", f"{stats['weighted_mean']:.4f}")
                    # Show prediction variance indicator
                    pred_range = stats['max'] - stats['min']
                    if pred_range < 0.01:
                        st.warning("⚠️ Low variance")
                    elif pred_range < 0.05:
                        st.info("ℹ️ Moderate variance")
                    else:
                        st.success("✅ Good variance")
            
            # Prediction distribution
            st.markdown("**Prediction Distribution Across Augmentations:**")
            # Create visualization as dictionary for line chart
            pred_dict = {f'A{i+1}': pred for i, pred in enumerate(stats['all_predictions'])}
            st.line_chart(pred_dict, height=200)
            st.caption(
                f"Range: {stats['min']:.4f} - {stats['max']:.4f} | "
                f"Mean: {stats['mean']:.4f} | Std: {stats['std']:.4f} | "
                f"Prediction Spread: {stats['max'] - stats['min']:.4f}"
            )
            
            # Debug info - check prediction variance
            pred_range = stats['max'] - stats['min']
            if pred_range < 0.01:
                st.error(
                    f"⚠️ **Critical Issue: Low Prediction Variance!**\n\n"
                    f"The model is producing nearly identical predictions across all {stats['num_augmentations']} augmentations:\n"
                    f"- Min: {stats['min']:.4f}\n"
                    f"- Max: {stats['max']:.4f}\n"
                    f"- Range: {pred_range:.4f} (only {pred_range*100:.2f}% variation)\n"
                    f"- Std Dev: {stats['std']:.4f}\n\n"
                    f"**This indicates the model is not differentiating between images.**\n\n"
                    f"**Possible causes:**\n"
                    f"1. Model may not be properly trained\n"
                    f"2. Model architecture issue\n"
                    f"3. Model expects different preprocessing\n"
                    f"4. Model outputs vegetation probability (inverted) - currently inverting output\n\n"
                    f"**Current fix:** Output is inverted (assuming model outputs vegetation prob). "
                    f"If predictions still don't vary, the model may need retraining."
                )
            elif pred_range < 0.05:
                st.warning(
                    f"⚠️ **Low prediction variance detected** (range: {pred_range:.4f}). "
                    f"Predictions are similar across augmentations, which may indicate the model "
                    f"needs better training or different preprocessing."
                )
            
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
