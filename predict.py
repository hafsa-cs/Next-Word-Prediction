# ==========================================
# NEXT WORD PREDICTION SYSTEM
# INFERENCE SCRIPT
# ==========================================

import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import MAX_SEQUENCE_LEN, MODEL_FOLDER, VOCAB_SIZE


# ==========================================
# LOAD ARTIFACTS
# ==========================================

print("Loading model and tokenizer...")

model = load_model(f"{MODEL_FOLDER}/best_model.keras")

with open(f"{MODEL_FOLDER}/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open(f"{MODEL_FOLDER}/max_len.pkl", "rb") as f:
    saved_max_len = pickle.load(f)

print("Ready for prediction!")


# ==========================================
# PREDICT NEXT WORDS
# ==========================================

def predict_next_words(seed_text, num_words=3):
    """
    Predict the next `num_words` after a given seed text.
    """
    text = seed_text.lower()

    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=saved_max_len - 1, padding="pre")

        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=-1)[0]

        # Skip if predicted index is out of vocabulary range
        if predicted_index >= VOCAB_SIZE or predicted_index == 0:
            output_word = "<OOV>"
        else:
            output_word = tokenizer.index_word.get(predicted_index, "<OOV>")

        text += " " + output_word

    return text


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    seed = "the future is"
    print("\nSeed Text :", seed)
    print("Prediction:", predict_next_words(seed, num_words=5))
