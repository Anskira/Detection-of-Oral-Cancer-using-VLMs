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

# Data Visualization
- Normal histopathological slide
  
  ![Normal_100x_13](https://github.com/user-attachments/assets/94945ec6-0eea-4d32-a250-b2ddb8d33820)

- OSCC
  
  ![OSCC_100x_1](https://github.com/user-attachments/assets/8af64d97-49e1-4fda-96b9-338488ce2a43)



### Running the training notebook
1. **Navigate to the Notebooks directory:**

   ```bash
   cd Notebooks
   ```

2. **Launch Jupyter Notebook:**

   ```bash
   jupyter training
   ```


