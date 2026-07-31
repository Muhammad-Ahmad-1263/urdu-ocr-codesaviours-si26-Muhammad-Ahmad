import streamlit as st
from transformers import RobertaTokenizer, ViTImageProcessor, TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

st.set_page_config(page_title="Urdu OCR -- Code Saviours SI-26")

# Set this to your HuggingFace MODEL repo id (not a Space), e.g. "Muhammad-Ahmad-1263/urdu-ocr-si26"
# Upload your model files there first (see deployment notes).
MODEL_PATH = "Muhammad-Ahmad-1263/urdu-ocr-si26"


@st.cache_resource
def load_model():
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    image_processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
    model.eval()
    return processor, model


processor, model = load_model()

st.title("Urdu OCR -- Code Saviours SI-26")
st.write("Upload an image containing Urdu text and get the extracted text.")

uploaded_file = st.file_uploader("Upload Urdu Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Extracting text..."):
        try:
            pixel_values = processor(image, return_tensors="pt").pixel_values
            with torch.no_grad():
                generated_ids = model.generate(pixel_values)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception as e:
            text = f"Something went wrong while processing this image: {e}"

    st.subheader("Extracted Urdu Text")
    st.write(text if text.strip() else "Could not extract any text from this image.")
else:
    st.info("Please upload an image to get started.")
