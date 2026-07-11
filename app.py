"""
FLASK WEB APP
=============
Ee app run chesthe, browser lo oka page open avutundi, akkada image upload
chesi "Generate Caption" click chesthe, trained model aa image ki caption
generate chesi chupistundi.

Run cheyadaniki mundu: train.py complete ayi, caption_model.h5 file undali.
"""

import pickle
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

print("Loading models... (idi konchem time padutundi, oka sari matrame)")

# Trained caption model (CNN features + LSTM decoder)
caption_model = load_model("caption_model.h5")

# Feature extractor (same InceptionV3 preprocess.py lo vadinatlu)
base_model = InceptionV3(weights="imagenet")
feature_extractor = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LENGTH = 34  # preprocess.py output nundi vachina max_length ikkada pettandi
index_to_word = {v: k for k, v in tokenizer.word_index.items()}


def extract_feature(img_path):
    image = load_img(img_path, target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return feature_extractor.predict(image, verbose=0)


def generate_caption(feature):
    """Greedy search: prathi step lo, most probable next word teesukoni,
    'endseq' వచ్చే varaku leda max_length varaku continue chestam."""
    in_text = "startseq"
    for _ in range(MAX_LENGTH):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=MAX_LENGTH)
        pred = caption_model.predict([feature, seq], verbose=0)
        pred_id = np.argmax(pred)
        word = index_to_word.get(pred_id)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    return caption


@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    if request.method == "POST":
        img_file = request.files["image"]
        img_path = "static_upload.jpg"
        img_file.save(img_path)
        feature = extract_feature(img_path)
        caption = generate_caption(feature)
    return render_template("index.html", caption=caption)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
