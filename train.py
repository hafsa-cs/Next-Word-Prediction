# ==========================================
# NEXT WORD PREDICTION SYSTEM
# TRAINING PIPELINE - FAST CPU MODE
# ==========================================

import os
import random
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from config import *
from preprocess import prepare_data


# ==========================================
# SET RANDOM SEED
# ==========================================

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


# ==========================================
# LOAD CLEAN DATA
# ==========================================

print("\n[1] Loading and Preprocessing Data...")

df = prepare_data()
texts = df["text"].values
print("Total Clean Samples:", len(texts))


# ==========================================
# TOKENIZER
# ==========================================

print("\n[2] Creating Tokenizer...")

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

total_words = VOCAB_SIZE + 1
print("Tokenizer unique words found:", len(tokenizer.word_index))
print("Model vocabulary size (fixed):", total_words)

os.makedirs(MODEL_FOLDER, exist_ok=True)
with open(f"{MODEL_FOLDER}/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)


# ==========================================
# CREATE SEQUENCES (FAST)
# ==========================================

print("\n[3] Creating Sequences...")

input_sequences = []
for line in texts:
    token_list = tokenizer.texts_to_sequences([line])[0]
    # Only generate sequences if we have enough tokens
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

print("Raw sequences generated:", len(input_sequences))

# CAP SEQUENCES FOR SPEED (FAST MODE)
if len(input_sequences) > MAX_TRAIN_SEQUENCES:
    random.shuffle(input_sequences)
    input_sequences = input_sequences[:MAX_TRAIN_SEQUENCES]
    print(f"Capped to {MAX_TRAIN_SEQUENCES} sequences for fast training.")


# ==========================================
# PADDING
# ==========================================

input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=MAX_SEQUENCE_LEN, padding="pre")
)

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

print("X shape:", X.shape)
print("y shape:", y.shape)


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE
)


# ==========================================
# MODEL ARCHITECTURE (FAST)
# ==========================================

print("\n[4] Building Model (Fast CPU Mode)...")

model = Sequential([
    Embedding(total_words, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LEN - 1),
    LSTM(LSTM_UNITS, return_sequences=True),
    Dropout(DROPOUT_RATE),
    LSTM(LSTM_UNITS // 2),
    Dropout(DROPOUT_RATE),
    Dense(DENSE_UNITS, activation="relu"),
    Dropout(DROPOUT_RATE),
    Dense(total_words, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
print("\n[OK] Model Built.")


# ==========================================
# CALLBACKS
# ==========================================

os.makedirs(GRAPH_FOLDER, exist_ok=True)

checkpoint_path = f"{MODEL_FOLDER}/best_model.keras"

checkpoint = ModelCheckpoint(
    checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1
)
early_stop = EarlyStopping(
    monitor="val_loss", patience=2, restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=1, verbose=1
)


# ==========================================
# TRAIN
# ==========================================

print("\n[5] Training Started (Fast Mode)...")
print(f"Steps per epoch: ~{len(X_train)//BATCH_SIZE}\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

print("\n[OK] Training Completed")


# ==========================================
# SAVE
# ==========================================

model.save(f"{MODEL_FOLDER}/final_model.keras")

with open(f"{MODEL_FOLDER}/max_len.pkl", "wb") as f:
    pickle.dump(MAX_SEQUENCE_LEN, f)


# ==========================================
# GRAPHS
# ==========================================

plt.figure(figsize=(10, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(f"{GRAPH_FOLDER}/accuracy.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f"{GRAPH_FOLDER}/loss.png")
plt.close()


print("\n[SUCCESS] ALL FILES SAVED")
print("[COMPLETE] PIPELINE COMPLETE")
