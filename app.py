import streamlit as st
import torch
import torch.nn as nn
import pickle
import os
from PIL import Image
import numpy as np
from torchvision import transforms

# Define the model architecture classes needed for loading the models
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )
    
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, n_residual_blocks=9, init_features=64):
        super(Generator, self).__init__()
        
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, init_features, 7),
            nn.InstanceNorm2d(init_features),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        in_features = init_features
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        
        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(init_features, output_channels, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

# Page configuration
st.set_page_config(
    page_title="Sketch-Photo Converter",
    page_icon="🖌️",
    layout="wide"
)

# App title and description
st.title("Sketch-Photo Converter")
st.write("Transform sketches into realistic photos and vice versa using CycleGAN")

# Function to load models
@st.cache_resource
def load_models():
    try:
        # Load sketch to photo model
        with open('exports/G_sketch2photo.pkl', 'rb') as f:
            sketch2photo = pickle.load(f)
        
        # Load photo to sketch model
        with open('exports/G_photo2sketch.pkl', 'rb') as f:
            photo2sketch = pickle.load(f)
            
        # Set models to evaluation mode
        sketch2photo.eval()
        photo2sketch.eval()
        
        return sketch2photo, photo2sketch
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None

# Function to process image
def process_image(model, image):
    # Define image transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Transform image to tensor
    input_tensor = transform(image).unsqueeze(0)
    
    # Perform inference
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # Convert output tensor to image
    output = output_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    output = ((output + 1) / 2 * 255).astype(np.uint8)
    
    return Image.fromarray(output)

# Load models
sketch2photo_model, photo2sketch_model = load_models()

if sketch2photo_model is not None and photo2sketch_model is not None:
    # Create two columns for sketch-to-photo and photo-to-sketch
    col1, col2 = st.columns(2)
    
    # Sketch to Photo
    with col1:
        st.subheader("Sketch to Photo")
        sketch_file = st.file_uploader("Upload a sketch", type=["jpg", "jpeg", "png"], key="sketch_uploader")
        
        if sketch_file is not None:
            sketch_image = Image.open(sketch_file).convert('RGB')
            st.image(sketch_image, caption="Input Sketch", use_column_width=True)
            
            if st.button("Generate Photo", key="generate_photo"):
                with st.spinner("Generating photo..."):
                    photo_image = process_image(sketch2photo_model, sketch_image)
                    st.image(photo_image, caption="Generated Photo", use_column_width=True)
                    
                    # Save image temporarily for download
                    photo_image.save("generated_photo.jpg")
                    with open("generated_photo.jpg", "rb") as file:
                        btn = st.download_button(
                            label="Download Photo",
                            data=file,
                            file_name="generated_photo.jpg",
                            mime="image/jpeg"
                        )
    
    # Photo to Sketch
    with col2:
        st.subheader("Photo to Sketch")
        photo_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"], key="photo_uploader")
        
        if photo_file is not None:
            photo_image = Image.open(photo_file).convert('RGB')
            st.image(photo_image, caption="Input Photo", use_column_width=True)
            
            if st.button("Generate Sketch", key="generate_sketch"):
                with st.spinner("Generating sketch..."):
                    sketch_image = process_image(photo2sketch_model, photo_image)
                    st.image(sketch_image, caption="Generated Sketch", use_column_width=True)
                    
                    # Save image temporarily for download
                    sketch_image.save("generated_sketch.jpg")
                    with open("generated_sketch.jpg", "rb") as file:
                        btn = st.download_button(
                            label="Download Sketch",
                            data=file,
                            file_name="generated_sketch.jpg",
                            mime="image/jpeg"
                        )
    
    # Add information in the footer
    st.divider()
    st.subheader("About this app")
    st.write("This app uses a CycleGAN model to convert between sketches and photos. The model was trained on a dataset of face sketches and photos.")
    
else:
    st.error("Failed to load models. Please make sure the model files exist in the 'exports' directory.")
    st.info("Expected paths: 'exports/G_sketch2photo.pkl' and 'exports/G_photo2sketch.pkl'")