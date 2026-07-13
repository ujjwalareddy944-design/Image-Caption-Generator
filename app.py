"""
FLASK WEB APP
=============
Running this app opens a page in your browser where you can upload an
image and click "Generate Caption" to see the trained model generate a
caption for it.

Before running this, train.py must have completed and caption_model.h5
must exist.
"""

import pickle
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

print("Loading models... (this takes a little time, only happens once)")

# Trained caption model (CNN features + LSTM decoder)
caption_model = load_model("caption_model.h5")

# Feature extractor (same InceptionV3 setup used in preprocess.py)
base_model = InceptionV3(weights="imagenet")
feature_extractor = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LENGTH = 34  # set this to the max_length value produced by preprocess.py
index_to_word = {v: k for k, v in tokenizer.word_index.items()}


def extract_feature(img_path):
    image = load_img(img_path, target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return feature_extractor.predict(image, verbose=0)


def generate_caption(feature):
    """Greedy search: at each step, pick the most probable next word,
    continuing until 'endseq' is produced or max_length is reached."""
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
