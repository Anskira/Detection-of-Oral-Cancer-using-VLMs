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

# Data Information
- Normal histopathological slide

  ![Normal_100x_13](https://github.com/user-attachments/assets/94945ec6-0eea-4d32-a250-b2ddb8d33820)

- OSCC

  ![OSCC_100x_1](https://github.com/user-attachments/assets/8af64d97-49e1-4fda-96b9-338488ce2a43)

## Data processing steps and image captioning
- Used the BLIP model to generate initial image captions.

- Cleaned the raw BLIP captions by fixing spacing issues and removing unwanted text.

- Split each caption into separate question and answer fields using string parsing.

- Removed entries where the split failed or the caption format was incorrect.

- Verified that every image had a valid caption and removed mismatched or missing pairs.

- Saved the filtered dataset in a clean CSV format for model training and evaluation.

# Model finetuning
- Loaded the Qwen2-VL-2B variant via Unsloth’s FastVisionModel API (the exact checkpoint used during experiments was the Unsloth Qwen2-VL instruct build). The model was run in quantized mode (4-bit) to fit GPU memory constraints.

- Converted the dataset to the Unsloth chat format required for training: each example became a messages entry where the user content is "<image>\n<question>" and images were passed separately (no nested {'type':'image'} in the text). This ensured token ↔ image alignment for the custom processor.

- Applied the same image preprocessing used during caption creation and training: convert to RGB, reduce resolution (resize), and load as PIL images before passing to the processor. Captions used BLIP-2 outputs framed as QA prompts (e.g., “Question: Explain why this image is classified as … Answer:”) and were cleaned/split into question/answer pairs prior to fine-tuning.

- Enabled LoRA-based PEFT with the same LoRA configuration used in experiments: r=16, lora_alpha=16, lora_dropout=0, bias="none", and use_rslora=False. Only LoRA layers (attention/MLP projection targets) were updated; the base model parameters remained frozen. Random seed and reproducibility parameters were set (seed 3407).

- Switched the model into training mode using FastVisionModel.for_training(...) and used Unsloth’s UnslothVisionDataCollator + trl.SFTTrainer for the optimization loop. Key training args that were used in runs:
   ```bash
   per_device_train_batch_size = 2
   
   gradient_accumulation_steps = 4 (effective batch size ≈ 8)
   
   warmup_steps = 5
   
   max_steps = 30 (short runs for quick iteration; longer runs possible)
   
   learning_rate = 2e-4
   
   optim = "adamw_8bit" (8-bit optimizer to reduce memory)
   
   weight_decay = 0.001
   
   lr_scheduler_type = "linear"
   
   max_length = 2048 (text tokens)
   
   seed = 3407
   ```
- output_dir set to a writable project path (avoid absolute root like /model)

- Used mixed strategies from Unsloth for memory and speed: 4-bit quantization for model weights, gradient checkpointing, and Unsloth’s runtime patches (xFormers when available) so training could run on a single T4 (or equivalent) GPU. Training logs showed ~28.9M trainable parameters (LoRA) out of ~2.24B total parameters.

- Per-step training loss and simple validation checks were logged. After training, the LoRA weights / PEFT adapter checkpoint was saved and the model was switched to inference mode with FastVisionModel.for_inference(...) for downstream evaluation and serving.

# Results & Discussion
- The performance of the model was evaluated before and after fine-tuning using BERTScore and validation accuracy. The improvement was clear across all metrics. Before fine-tuning, the model produced reasonable explanations, but the precision, recall, and F1 scores indicated inconsistency in how well the generated answers aligned with the ground-truth captions. After fine-tuning on the BLIP-generated QA dataset, all three BERTScore metrics increased sharply, reaching approximately 0.96, demonstrating that the model learned to produce more accurate and context-aligned explanations.

- Validation accuracy reached 100% in both cases. This result reflects that the classification portion of the task was straightforward for the model even in its pre-trained state, while the fine-tuning primarily improved the quality and specificity of textual explanations, not the label prediction itself. The model also showed reduced variability in its outputs after fine-tuning and generated more consistent reasoning aligned with the domain-specific patterns in oral histopathology images.

- Overall, the results confirm that fine-tuning was effective in strengthening the explanation-generation ability of the model without degrading classification performance.

  <img width="845" height="165" alt="image" src="https://github.com/user-attachments/assets/707674a0-58c2-4994-87ca-16fa39165ef5" />

# User Interface
This is how the UI looks like and I have also added a demonstration video in the 'Demonstration' folder
<img width="1127" height="322" alt="UI" src="https://github.com/user-attachments/assets/ee633bf8-01fb-4b99-94dd-f44353ca2d45" />

### Running the training notebook
1. **Navigate to the Notebooks directory:**

   ```bash
   cd Notebooks
   ```

2. **Launch Jupyter Notebook:**

   ```bash
   jupyter training
   ```


