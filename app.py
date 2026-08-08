import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pytesseract

st.set_page_config(page_title="Urdu OCR -- Code Saviours SI-26")


def deskew(gray: np.ndarray) -> np.ndarray:
    """Estimate and correct small rotation/skew using the text's bounding box angle."""
    # Threshold so text pixels are white on a black background
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 20:
        return gray  # not enough signal to safely estimate an angle
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    # Skip correction for negligible or clearly-wrong angle estimates
    if abs(angle) < 0.5 or abs(angle) > 15:
        return gray
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def upscale_to_target_dpi(gray: np.ndarray, target_long_side: int = 2000) -> np.ndarray:
    """Tesseract performs best around 300 DPI; upscale small images accordingly."""
    h, w = gray.shape[:2]
    long_side = max(h, w)
    if long_side >= target_long_side:
        return gray
    scale = target_long_side / long_side
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def preprocess(pil_image: Image.Image) -> np.ndarray:
    """Grayscale -> upscale (~300 DPI equivalent) -> denoise -> deskew."""
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = upscale_to_target_dpi(gray)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    gray = deskew(gray)
    return gray


def extract_urdu_text(pil_image: Image.Image) -> str:
    processed = preprocess(pil_image)
    text = pytesseract.image_to_string(processed, lang="urd", config="--psm 6")
    return text.strip()


st.title("Urdu OCR -- Code Saviours SI-26")
st.caption("Classical OCR pipeline (Tesseract + preprocessing) — see README for the TrOCR research findings.")
st.write("Upload an image containing Urdu text and get the extracted text.")

uploaded_file = st.file_uploader("Upload Urdu Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Preprocessing and extracting text..."):
        try:
            text = extract_urdu_text(image)
        except Exception as e:
            text = f"Something went wrong while processing this image: {e}"

    st.subheader("Extracted Urdu Text")
    st.write(text if text else "Could not extract any text from this image.")
else:
    st.info("Please upload an image to get started.")
