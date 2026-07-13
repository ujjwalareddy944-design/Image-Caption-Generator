"""
MODEL ARCHITECTURE
==================
This is an "Encoder-Decoder" architecture:

- ENCODER: Pretrained InceptionV3 (already run in preprocess.py — feature
  vectors are ready). In this model file, we compress that 2048-dim
  vector down to 256 dimensions using a Dense layer.

- DECODER: An LSTM (Long Short-Term Memory) network. It looks at the
  image features plus the words generated so far, and predicts the next
  word — one word at a time, until the full caption is generated.

Diagram (simplified):
  Image -> InceptionV3 (2048) -> Dense(256) ---\
                                                  +--> Add -> Dense -> LSTM -> next word
  Partial caption -> Embedding -> LSTM(256) ----/
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add


def build_caption_model(vocab_size, max_length, feature_dim=2048, embedding_dim=256, lstm_units=256):
    # --- Image feature branch ---
    inputs1 = Input(shape=(feature_dim,), name="image_features")
    fe1 = Dropout(0.4)(inputs1)
    fe2 = Dense(embedding_dim, activation="relu")(fe1)

    # --- Text sequence branch ---
    inputs2 = Input(shape=(max_length,), name="caption_sequence")
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.4)(se1)
    se3 = LSTM(lstm_units)(se2)

    # --- Combine both branches ---
    decoder1 = add([fe2, se3])
    decoder2 = Dense(lstm_units, activation="relu")(decoder1)
    outputs = Dense(vocab_size, activation="softmax")(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model
