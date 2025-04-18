# FaceSketchGAN

A CycleGAN-based deep learning model for bidirectional conversion between face sketches and photographs, with a user-friendly Streamlit interface.

## Overview

FaceSketchGAN uses a CycleGAN architecture to perform unpaired image-to-image translation between sketches and photographs. The model can convert hand-drawn sketches into photorealistic face images, and transform photographs into sketch-like images.

Key features:
- Bidirectional translation (sketch→photo and photo→sketch)
- No need for paired training data
- User-friendly Streamlit web interface
- Easy-to-use download functionality for generated images

## Installation

### Prerequisites
- Python 3.7+
- PyTorch 1.8+
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository
```bash
git clone https://github.com/hassansaeed-isb/FaceSketchGAN.git
cd FaceSketchGAN
```

2. Create and activate a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Web Interface

The simplest way to use FaceSketchGAN is through the Streamlit web interface:

```bash
streamlit run app.py
```

The app should open automatically in your web browser. If not, navigate to http://localhost:8501.

### Training a Model

To train the CycleGAN on your own dataset:

```bash
python train.py --sketch_dir path/to/sketches --photo_dir path/to/photos --epochs 100
```

Additional training arguments:
- `--batch_size`: Batch size for training (default: 4)
- `--img_size`: Size of input images (default: 256)
- `--lr`: Learning rate (default: 0.0002)
- `--max_images`: Maximum number of images to use from each domain (default: 500)

### Testing the Model

To test the model on new images:

```bash
python test.py --model_path exports/G_sketch2photo.pkl --input_dir path/to/test/sketches --output_dir path/to/results
```

## Model Architecture

FaceSketchGAN implements a CycleGAN with:
- Two generators (sketch→photo and photo→sketch) with ResNet-based architecture
- Two PatchGAN discriminators 
- Cycle consistency loss to maintain content consistency
- Identity loss to preserve color composition

## Performance

The model was evaluated using standard image quality metrics:
- Average PSNR: 10.81
- Average SSIM: 0.2225

These scores reflect the challenging nature of unpaired image-to-image translation. Qualitative results show good visual quality despite the relatively low numerical scores.


## Acknowledgments

- [CycleGAN Paper](https://arxiv.org/abs/1703.10593) by Jun-Yan Zhu et al.
- [PyTorch](https://pytorch.org/)
- [Streamlit](https://streamlit.io/) for the web interface
