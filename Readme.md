# Introduction
The objective of this project is to fine-tune a multimodal
vision-language model, Qwen2-VL-2B, for the classification of oral histopathological images into Normal and Oral
Squamous Cell Carcinoma (OSCC) categories. Unlike traditional convolutional neural networks (CNNs), Qwen2-VL2B integrates both visual and textual modalities, allowing for
joint reasoning over image content and associated medical
context. The model is fine-tuned using a curated dataset of
histopathological slides, enabling it to understand complex
tumor morphology and associate it with descriptive textual
cues. This work aims to improve diagnostic accuracy and
interpretability in digital pathology applications, contributing
toward clinically reliable multimodal AI systems

# Getting Started

## Prerequisites
- Python 3.7+
- pip (Python Package Installer)


# Getting Started

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Clone the Repository
```bash
git clone https://github.com/Anskira/Detection-of-Oral-Cancer-using-VLMs.git
```

### Install the required packages
```bash
   pip install -r requirements.txt
```

### Running the training notebook
1. **Navigate to the Notebooks directory:**

   ```bash
   cd Notebooks
   ```

2. **Launch Jupyter Notebook:**

   ```bash
   jupyter training
   ```


