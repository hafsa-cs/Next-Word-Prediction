# ===============================
# PROJECT CONFIGURATION
# Next Word Prediction System
# FAST MODE FOR CPU / DEMO
# ===============================

# Dataset Path
DATASET_PATH = "dataset/WikiQuotes.csv"

# Use only quote_text column
TEXT_COLUMN = "quote_text"

# Maximum rows to load (5k is plenty for a solid next-word model)
MAX_ROWS = 5000

# Maximum words in a sentence (pad/truncate to this)
MAX_SEQUENCE_LEN = 15

# Vocabulary Size
VOCAB_SIZE = 5000

# Embedding Dimension (small = fast)
EMBEDDING_DIM = 50

# LSTM Units (small = fast)
LSTM_UNITS = 32

# Dense Layer Units
DENSE_UNITS = 64

# Dropout
DROPOUT_RATE = 0.2

# Training
EPOCHS = 8
BATCH_SIZE = 1024

# Cap total training sequences (safety valve for speed)
MAX_TRAIN_SEQUENCES = 100000

# Validation Split
VALIDATION_SPLIT = 0.2

# Random Seed
RANDOM_STATE = 42

# Output folders
MODEL_FOLDER = "models"
GRAPH_FOLDER = "graphs"
