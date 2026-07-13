# Image Caption Generator (Generative AI / Deep Learning)

## Project Overview

This project generates a natural-language caption for a given image (e.g.
"a dog running on the grass").

**Architecture — Encoder-Decoder:**
- **Encoder (CNN):** A pretrained **InceptionV3** model (already trained
  on ImageNet) is used to convert each image into a 2048-dimensional
  feature vector. This is **transfer learning** — we reuse an already
  trained model to "understand" the image, instead of training a CNN
  from scratch.
- **Decoder (LSTM):** An LSTM (a type of RNN) takes the image feature
  vector plus the words generated so far, and predicts the next word.
- **Dataset:** Flickr8k (8,000 images, each with 5 different human-written
  captions)

## Files

| File | Purpose |
|---|---|
| `download_data.py` | Instructions for downloading the dataset |
| `preprocess.py` | Converts images -> features.pkl, captions -> tokenizer.pkl |
| `model.py` | Defines the CNN + LSTM model architecture |
| `train.py` | Trains the model and saves `caption_model.h5` |
| `app.py` | Flask web app — upload an image, get a caption |
| `templates/index.html` | Web page UI |

## How to Run — Step by Step

### 1. Install dependencies
```bash
git clone <your-repo-url>
cd image-caption-generator
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download the dataset
Open `download_data.py` for full instructions, download the **Flickr8k**
dataset from Kaggle, and place it so you have:
```
data/Images/          <- 8000+ .jpg files
data/captions.txt
```
Verify it's in place:
```bash
python download_data.py
```

### 3. Run preprocessing
```bash
python preprocess.py
```
This will:
- Clean the captions and build a tokenizer
- Convert every image into a feature vector using InceptionV3
  (~30-60 minutes on CPU for 8000 images, much faster on GPU)

Output: `features.pkl`, `captions_cleaned.pkl`, `tokenizer.pkl`

### 4. Train the model
```bash
python train.py
```
This trains for 20 epochs by default. **Training will be very slow
without a GPU** — it's recommended to run this on **Google Colab**
(free GPU access).
Output: `caption_model.h5`

### 5. Run the web app
```bash
python app.py
```
Open `http://localhost:5000` in your browser, upload an image, and
generate a caption.

## Key Points to Mention in an Interview

- "I used transfer learning — a pretrained InceptionV3 as a feature
  extractor, so I didn't need to train a CNN from scratch."
- "The LSTM decoder looks at the previous words plus the image features
  to predict the next word, one word at a time, until the full sentence
  is generated."
- "Model quality can be evaluated using the BLEU score — a metric that
  measures how closely the generated caption matches human-written
  reference captions."
- "A real-world challenge is training time (a GPU really helps), and
  model complexity increases as the vocabulary size grows."

## .gitignore suggestion

If you're uploading this to GitHub, avoid committing large generated
files. Create a `.gitignore` with:
```
venv/
data/
*.pkl
*.h5
__pycache__/
static_upload.jpg
```
