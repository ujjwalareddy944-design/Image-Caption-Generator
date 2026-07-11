# Image Caption Generator (Generative AI / Deep Learning)

## Idi enti (project explanation)

Ee project oka image icchite, aa image ki matching natural language caption (e.g.
"a dog running on the grass") generate chestundi.

**Architecture — "Encoder-Decoder":**
- **Encoder (CNN):** Pretrained **InceptionV3** model (ImageNet meeda already
  train ayindi) vadi, image ni 2048 numbers (feature vector) ga convert
  chestam. Idi **transfer learning** — manam CNN ni malli train cheyము, already
  nerchukunna model ni "image ni ardham chesukovadaniki" vadatam matrame.
- **Decoder (LSTM):** Ee feature vector + ippati varaku generate chesina words
  teesukoni, next word ento predict chese LSTM (RNN type) network.
- **Dataset:** Flickr8k (8,000 images, prathi image ki 5 different captions)

## Files enti

| File | Pని |
|---|---|
| `download_data.py` | Dataset ela download cheyalo instructions |
| `preprocess.py` | Images -> features.pkl, captions -> tokenizer.pkl |
| `model.py` | CNN+LSTM model architecture define chestundi |
| `train.py` | Model ni train chesi `caption_model.h5` save chestundi |
| `app.py` | Flask web app — image upload chesi caption chudochu |
| `templates/index.html` | Web page UI |

## Step-by-step: Ela run cheyali

### 1. Install cheyandi
```bash
cd 1-image-caption-generator
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dataset download cheyandi
`download_data.py` file open chesi, akkada unna instructions follow chesi
Flickr8k dataset download chesi, `data/Images/` and `data/captions.txt` ee
folder lo pettandi. (Check cheyadaniki: `python download_data.py`)

### 3. Preprocessing run cheyandi
```bash
python preprocess.py
```
Idi chestundi:
- Captions clean chesi tokenizer build chestundi
- Prathi image ni InceptionV3 tho feature vector ga convert chestundi
  (8000 images ki ~30-60 min padochu CPU meeda, GPU unte fast)

Output: `features.pkl`, `captions_cleaned.pkl`, `tokenizer.pkl`

### 4. Model train cheyandi
```bash
python train.py
```
Idi 20 epochs training chestundi (default). **GPU లేకపోతే idi చాలా slow
avvachu** — recommend chesedi: **Google Colab** (free GPU) lo run cheyadam.
Output: `caption_model.h5`

### 5. Web app run cheyandi
```bash
python app.py
```
Browser lo `http://localhost:5000` open chesi, image upload chesi caption
generate cheyandi.

## Interview lo cheppadaniki key points

- "Transfer learning vadanu — pretrained InceptionV3 ni feature extractor ga
  vadanu, so nenu CNN ni scratch nundi train cheyalsina అవసరం లేదు."
- "LSTM decoder, previous words + image features chusi next word predict
  chestundi, oka word tarwata oka word ga full sentence generate ayye varaku."
- "Model quality ni BLEU score tho evaluate cheyachu — generated caption,
  actual human captions తో ఎంత match avutundo measure chese metric."
- "Real-world challenge: training time chala ekkuva (GPU కావాలి), mariyu
  vocabulary size peruguthunna కొద్దీ model complexity peruguthundi."
