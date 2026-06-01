# 🖼️ Image Description using Vision-Language Models (VLMs)

A research project comparing how well different AI vision models can describe images — tested and evaluated during an internship.

---

## What's this about?

The goal was simple: give an image to several state-of-the-art vision-language models (VLMs) and see which one generates the most accurate, human-like caption. We ran each model on the same dataset of 150 images and measured their performance using standard NLP evaluation metrics.

Think of it as a head-to-head race between some of the best open-source image captioning models out there.

---

## Models Compared

| Model | Who made it |
|---|---|
| **BLIP** | Salesforce |
| **SigLIP** | Google |
| **GIT** (Generative Image-to-Text) | Microsoft |

---

## How it works

1. Load 150 images from the [`jaimin/Image_Caption`](https://huggingface.co/datasets/jaimin/Image_Caption) dataset on HuggingFace.
2. Pass each image through all three models to generate a caption.
3. Compare the generated captions against the ground-truth captions using three metrics:
   - **BLEU** — how much n-gram overlap there is between the generated and reference text
   - **ROUGE** (ROUGE-1 & ROUGE-L) — recall-oriented overlap scoring
   - **METEOR** — a more flexible metric that accounts for synonyms and stemming

---

## Files in this repo

├── Blip-Siglip-Git.py       # Main script: runs all 3 models and evaluates them

├── Vit-gpt2.py              # Extra experiment with ViT + GPT-2 combo

├── llava_(1).py             # LLaVA model experiment

├── qwen_vl_(1).py           # Qwen-VL model experiment

├── Internship_Report.pdf    # Full internship report with findings

└── VLMs-paper.docx          # Research paper write-up

---

## Setup & Requirements

This project was built in Google Colab. To run it yourself:

```bash
pip install transformers datasets torch pillow nltk rouge-score
```

Then download NLTK data inside Python:

```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## Quick Start

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image = Image.open("your_image.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
print(caption)
```

---

## Tech Stack

- Python
- HuggingFace `transformers` & `datasets`
- PyTorch
- NLTK, rouge-score
- Google Colab

---

## Notes

- All experiments were run on 150 training samples due to compute constraints.
- SigLIP computes image-text similarity scores (cosine similarity between embeddings) rather than generating captions directly.
- Full analysis and metric results are in the internship report PDF.
