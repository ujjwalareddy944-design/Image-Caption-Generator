"""
TRAINING SCRIPT
===============
Once preprocess.py has run (features.pkl, captions_cleaned.pkl,
tokenizer.pkl are ready), this script trains the model.

Training is much faster with a GPU (e.g., free Google Colab GPU). It will
also run on CPU, just slower.
"""

import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from model import build_caption_model


def load_data():
    with open("features.pkl", "rb") as f:
        features = pickle.load(f)
    with open("captions_cleaned.pkl", "rb") as f:
        mapping = pickle.load(f)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return features, mapping, tokenizer


def data_generator(mapping, features, tokenizer, max_length, vocab_size, batch_size=32):
    """Loading all captions into memory at once would use too much RAM,
    so we generate data batch by batch (standard Keras generator pattern)."""
    while True:
        X1, X2, y = [], [], []
        count = 0
        for img_name, captions in mapping.items():
            if img_name not in features:
                continue
            feature = features[img_name][0]
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)
            count += 1
            if count == batch_size:
                yield (np.array(X1), np.array(X2)), np.array(y)
                X1, X2, y = [], [], []
                count = 0


if __name__ == "__main__":
    print("Loading preprocessed data...")
    features, mapping, tokenizer = load_data()

    vocab_size = min(5000, len(tokenizer.word_index) + 1)
    all_captions = [c for caps in mapping.values() for c in caps]
    max_length = max(len(c.split()) for c in all_captions)

    print(f"Vocab size: {vocab_size}, Max length: {max_length}, Total images: {len(mapping)}")

    print("Building model...")
    model = build_caption_model(vocab_size, max_length)
    model.summary()

    EPOCHS = 20
    BATCH_SIZE = 32
    steps = len(mapping) // BATCH_SIZE

    print(f"\nStarting training: {EPOCHS} epochs, {steps} steps/epoch")
    generator = data_generator(mapping, features, tokenizer, max_length, vocab_size, BATCH_SIZE)

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        model.fit(generator, steps_per_epoch=steps, epochs=1, verbose=1)
        model.save("caption_model.h5")  # save after every epoch

    print("\n🎉 Training complete! caption_model.h5 has been saved. You can now run app.py.")
