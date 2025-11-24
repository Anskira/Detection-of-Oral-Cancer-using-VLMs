import gradio as gr
import torch
from PIL import Image
from unsloth import FastVisionModel

model, processor = FastVisionModel.from_pretrained(
    'model/qwen2vl_lora',
    load_in_4bit=True
)

from PIL import Image

def convert_to_rgb(image):
    if image.mode == "RGB":
        return image
    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    composite = Image.alpha_composite(background, image_rgba)
    return composite.convert("RGB")

def reduce_image_size(image, scale=0.5):
    w, h = image.size
    return image.resize((int(w*scale), int(h*scale)))

def preprocess_image(image):
    image = convert_to_rgb(image)
    image = reduce_image_size(image, scale=0.5)
    return image




FastVisionModel.for_inference(model)

label_map = {
    0: "Normal",
    1: "OSCC"
}

def extract_generated_answer(decoded_text: str) -> str:
    """
    Extracts only the model-generated assistant response
    from the decoded output.
    """
    # split by 'assistant' (last occurrence)
    parts = decoded_text.strip().split("assistant")
    if len(parts) < 2:
        return decoded_text.strip()  # fallback
    
    answer = parts[-1].strip()  # text after the last 'assistant'
    return answer


def predict(i: Image.Image):
    image = preprocess_image(i)
    instruction = "Explain whether the histopathological image of Oral cavity is Normal or OSCC"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": instruction
                },
            ]
        }
    ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt = True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
    )

    device = model.device
    for k,v in list(inputs.items()):
        if torch.is_tensor(v):
            inputs[k] = v.to(device)
        elif isinstance(v, list):
            inputs[k] = [t.to(device) if torch.is_tensor(t) else t for t in v]

    with torch.inference_mode():
        result_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            use_cache=True,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )

    pred_text = processor.batch_decode(result_ids, skip_special_tokens=True)[0]
    pred_text = extract_generated_answer(pred_text)
    # print(pred_text, t)
    return pred_text
    

#Gradio UI

title = "🧬 Histopathology Slide Classifier (Fine-Tuned Model)"
description = """
Upload a **histopathological image**, and the fine-tuned model will classify it.<br>
Model is optimized on your custom dataset and provides confidence scores.
"""


interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Histopathology Slide"),
    outputs="text",
    title=title,
    description=description,
    examples=[
        ["example1.png"],
        ["example2.png"]
    ],
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7861, share= True, debug = True)