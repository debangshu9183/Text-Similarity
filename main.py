import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
from datasets import load_dataset
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# Download nltk data
nltk.download('punkt')

# =======================
# Load and Prepare Dataset
# =======================
# In Colab, you may use:
# from google.colab import files
# files.upload()

df = pd.read_csv("questions.csv.zip")
print(df.head())

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

texts1 = df['question1'].astype(str).tolist()
texts2 = df['question2'].astype(str).tolist()
labels = df['is_duplicate'].astype(int).values

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts1 + texts2)

seq1 = tokenizer.texts_to_sequences(texts1)
seq2 = tokenizer.texts_to_sequences(texts2)

maxlen = 50
X1 = pad_sequences(seq1, maxlen=maxlen, padding='post')
X2 = pad_sequences(seq2, maxlen=maxlen, padding='post')

X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
    X1, X2, labels, test_size=0.2, random_state=42
)

# =======================
# Define Transformer Components
# =======================
from tensorflow.keras.layers import (
    Layer, Dense, Embedding, LayerNormalization, Dropout,
    Input, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model

class PositionalEncoding(Layer):
    def __init__(self, maxlen, d_model):
        super().__init__()
        self.pos_encoding = self.get_positional_encoding(maxlen, d_model)

    def get_positional_encoding(self, maxlen, d_model):
        angle_rads = np.arange(maxlen)[:, np.newaxis] / np.power(
            10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]


class TransformerBlock(Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, training=False):
        attn_output = self.att(x, x)
        out1 = self.layernorm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))

# =======================
# Model Building
# =======================
def build_encoder(vocab_size, maxlen, d_model, num_heads, dff):
    inputs = Input(shape=(maxlen,))
    x = Embedding(vocab_size, d_model)(inputs)
    x = PositionalEncoding(maxlen, d_model)(x)
    x = TransformerBlock(d_model, num_heads, dff)(x)
    x = GlobalAveragePooling1D()(x)
    return Model(inputs, x, name="shared_encoder")

def build_similarity_model(vocab_size, maxlen, d_model, num_heads, dff):
    encoder = build_encoder(vocab_size, maxlen, d_model, num_heads, dff)

    input1 = Input(shape=(maxlen,))
    input2 = Input(shape=(maxlen,))

    encoded1 = encoder(input1)
    encoded2 = encoder(input2)

    cosine = tf.keras.layers.Dot(axes=1, normalize=True)([encoded1, encoded2])
    output = Dense(1, activation='sigmoid')(cosine)

    model = Model(inputs=[input1, input2], outputs=output)
    return model, encoder

# =======================
# Compile and Train
# =======================
model, encoder = build_similarity_model(
    vocab_size=10000,
    maxlen=50,
    d_model=64,
    num_heads=4,
    dff=128
)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

hist = model.fit(
    [X1_train, X2_train], y_train,
    validation_data=([X1_val, X2_val], y_val),
    epochs=15,
    batch_size=64
)

# =======================
# Evaluate
# =======================
loss, acc = model.evaluate([X1_val, X2_val], y_val)
print(f"Validation Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# =======================
# Plot Accuracy & Loss
# =======================
plt.plot(hist.history['accuracy'], label='Train Accuracy')
plt.plot(hist.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(hist.history['loss'], label='Train Loss')
plt.plot(hist.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# =======================
# Test on Custom Questions
# =======================
q1 = "How to invest in stock market?"
q2 = "What are the steps to invest in shares?"

seq1 = tokenizer.texts_to_sequences([q1])
seq2 = tokenizer.texts_to_sequences([q2])
pad1 = pad_sequences(seq1, maxlen=maxlen, padding='post')
pad2 = pad_sequences(seq2, maxlen=maxlen, padding='post')

pred = model.predict([pad1, pad2])
print("Similarity score:", pred[0][0])
