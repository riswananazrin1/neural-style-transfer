# Neural Style Transfer and Sketch Generation

## Overview
This is a group academic project developed as part of the Computer Vision subject. It implements Neural Style Transfer using a pretrained VGG-19 model and sketch generation using OpenCV.

The project demonstrates how deep learning and image processing techniques can be used to create artistic image outputs.

## Features

- **Neural Style Transfer:** Uses a pretrained VGG-19 model to preserve image content while transferring artistic style from a separate style image.
- **Sketch Generation:** Converts the original content image into a sketch-like image using OpenCV image processing techniques.
- **Image Processing:** Resizes and preprocesses images before generating the outputs.

## Technologies Used

- Python
- TensorFlow and Keras
- VGG-19 pretrained on ImageNet
- NumPy
- Pillow (PIL)
- OpenCV

## Project Structure

```text
neural-style-transfer/
├── main.py
├── content_img/
│   └── content.jpg
├── style_img/
│   └── style.jpg
├── outputs/
│   ├── stylized_result.jpg
│   └── sketch_result.jpg
├── requirements.txt
└── README.md
```

## How It Works

### 1. Neural Style Transfer
The program extracts content and style features using VGG-19. It optimizes a generated image using content loss, style loss, and total variation loss to produce a stylized result.

### 2. Sketch Generation
The program converts the original content image to grayscale, inverts it, applies Gaussian blur, and performs image division to create a sketch-like result.

## Installation

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

## How to Run

1. Place the content image at `content_img/content.jpg`.
2. Place the style image at `style_img/style.jpg`.
3. Ensure the `outputs` folder exists.
4. Run the following command:

```bash
python main.py
```

## Output

- `outputs/stylized_result.jpg` — Neural Style Transfer result.
- `outputs/sketch_result.jpg` — Sketch generated from the original content image.

## Objective

To explore deep learning-based feature extraction and image processing techniques for artistic image transformation.

## Academic Context

Developed as a group academic project for the Computer Vision subject.
