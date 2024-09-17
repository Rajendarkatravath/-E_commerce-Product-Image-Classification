import streamlit as st
from transformers import pipeline
from PIL import Image
import torch
import os



# App Title
st.title("E-commerce Product Image Classification with GPU Support")
# Hugging Face API login
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Option to manually enter Hugging Face API key
if not HUGGINGFACE_API_KEY:
    HUGGINGFACE_API_KEY = st.text_input("Enter your Hugging Face API Key", type="password")

# Function to load model from Hugging Face pipeline with GPU support
@st.cache_data
def load_model(api_key):
    # Login to Hugging Face
    from huggingface_hub import login
    login(api_key)

    # Check if CUDA (GPU) is available and set device
    device = 0 if torch.cuda.is_available() else -1
    
    # Load image classification pipeline with GPU support if available
    classifier = pipeline("image-classification", model="google/vit-base-patch16-224", device=device)
    
    return classifier

# If API key is provided, load the model
if HUGGINGFACE_API_KEY:
    classifier = load_model(HUGGINGFACE_API_KEY)
    st.success("Hugging Face model loaded successfully!")

    # Upload image
    uploaded_file = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open image and display
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Product Image", use_column_width=True)

        # Classify image using the Hugging Face pipeline
        with st.spinner("Classifying..."):
            results = classifier(image)
        
        # Display the results
        st.write("## Classification Results:")
        for result in results:
            st.write(f"**Label**: {result['label']}, **Confidence**: {result['score']:.4f}")
else:
    st.warning("Please enter your Hugging Face API key to use the model.")
