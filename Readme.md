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
git clone https://github.com/UF-EEE5776-Spring25/project-1-AnshulPatil2911.git
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

3. Change the path to the csv file in the config.yaml file as per your requirements

4. Provide the path of the directory in which you pipeline .pkl files stored as per your requirements.

5. Provide the path of the directory in which all the model's .pkl file are stored as per your requirements
