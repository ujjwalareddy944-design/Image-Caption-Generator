"""
PREPROCESSING STEP
==================
Ee script rendu pనులు chestundi:
1. Prathi image ni pretrained InceptionV3 (CNN) tho pass chesi, ఒక feature
   vector (1x2048) extract chestundi -> features.pkl file lo save chestundi
   (idi "transfer learning" -> already trained CNN vadi image ni ardham
   chesukovadam, malli manam CNN train cheyakunda).
2. Captions text ni chadivi, clean chesi (lowercase, punctuation remove),
   tokenizer build chesi, sequences ga convert chestundi.

Idi oka sari matrame run cheyali (training ki mundu). Output:
   - features.pkl   (image_name -> 2048-dim vector)
   - captions_cleaned.pkl (image_name -> list of cleaned captions)
   - tokenizer.pkl  (word <-> index mapping)
"""

import os
import pickle
import string
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer

IMAGES_DIR = "data/Images"
CAPTIONS_FILE = "data/captions.txt"


def extract_image_features():
    """Pretrained InceptionV3 (ImageNet meeda train ayindi) vadi, last classification
    layer teesi vesi, image ni 2048-dim feature vector ga convert chestam."""
    print("Loading InceptionV3 (pretrained on ImageNet)...")
    base_model = InceptionV3(weights="imagenet")
    # last layer (classification) remove chesi, second-to-last layer output teesukuntam
    model = base_model
    from tensorflow.keras.models import Model
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

    features = {}
    image_files = os.listdir(IMAGES_DIR)
    print(f"{len(image_files)} images meeda feature extraction start avutundi...")

    for img_name in tqdm(image_files):
        img_path = os.path.join(IMAGES_DIR, img_name)
        image = load_img(img_path, target_size=(299, 299))  # InceptionV3 needs 299x299
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        features[img_name] = feature

    with open("features.pkl", "wb") as f:
        pickle.dump(features, f)
    print("✅ features.pkl save chesamu.")
    return features


def clean_captions():
    """captions.txt chadivi -> lowercase, punctuation remove, 'startseq'/'endseq'
    tags add chesi -> per-image caption list build chestam."""
    with open(CAPTIONS_FILE, "r") as f:
        lines = f.readlines()[1:]  # first line header, skip

    mapping = {}
    table = str.maketrans("", "", string.punctuation)

    for line in lines:
        parts = line.strip().split(",", 1)
        if len(parts) != 2:
            continue
        img_name, caption = parts
        caption = caption.lower().translate(table)
        caption = " ".join([w for w in caption.split() if w.isalpha() and len(w) > 1])
        caption = "startseq " + caption + " endseq"
        mapping.setdefault(img_name, []).append(caption)

    with open("captions_cleaned.pkl", "wb") as f:
        pickle.dump(mapping, f)
    print(f"✅ captions_cleaned.pkl save chesamu. Total images: {len(mapping)}")
    return mapping


def build_tokenizer(mapping, vocab_size=5000):
    """Anni captions kalipi, most common 'vocab_size' words tho tokenizer build
    chestam (word -> integer index mapping, model ki text ardham cheyadaniki)."""
    all_captions = [c for caps in mapping.values() for c in caps]
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<unk>")
    tokenizer.fit_on_texts(all_captions)

    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    max_length = max(len(c.split()) for c in all_captions)
    print(f"✅ tokenizer.pkl save chesamu. Vocab size: {vocab_size}, Max caption length: {max_length}")
    return tokenizer, max_length


if __name__ == "__main__":
    print("Step 1/3: Cleaning captions...")
    mapping = clean_captions()

    print("\nStep 2/3: Building tokenizer...")
    build_tokenizer(mapping)

    print("\nStep 3/3: Extracting image features (idi konchem time padutundi)...")
    extract_image_features()

    print("\n🎉 Preprocessing complete! Ippudu train.py run cheyachu.")
