# Next Word Prediction using Deep Learning (Bi-LSTM)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-LSTM-green)
![NLP](https://img.shields.io/badge/NLP-Text%20Generation-purple)
![License](https://img.shields.io/badge/License-MIT-red)

## Project Overview

This project implements a **Next Word Prediction System** using **Natural Language Processing (NLP)** and **Deep Learning**. The model learns language patterns from a large collection of quotes and predicts the most likely next word based on user input.

The project demonstrates the complete machine learning workflow including data preprocessing, exploratory data analysis (EDA), sequence generation, model training, evaluation, and interactive prediction.

---

## Features

- Large-scale text preprocessing
- Text cleaning and normalization
- Word Tokenization
- Sequence Generation
- Padding of input sequences
- Bidirectional LSTM Neural Network
- Early Stopping to reduce overfitting
- Interactive Next Word Prediction
- Accuracy & Loss Visualization
- Word Frequency Analysis
- Word Cloud Generation
- Saved trained model for future predictions

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- WordCloud
- Pickle

---

## Project Structure

```
Next-Word-Prediction/
│
├── dataset/
│   └── WikiQuotes.csv
│
├── graphs/
│   ├── accuracy.png
│   ├── loss.png
│   ├── top_words.png
│   └── wordcloud.png
│
├── models/
│   ├── best_model.keras
│   ├── tokenizer.pkl
│   └── max_len.pkl
│
├── config.py
├── preprocess.py
├── train.py
├── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Workflow

1. Load Dataset
2. Clean Text
3. Remove Missing Values
4. Remove Duplicate Records
5. Generate Word Frequency Analysis
6. Generate Word Cloud
7. Tokenize Text
8. Generate Training Sequences
9. Apply Padding
10. Train Bidirectional LSTM Model
11. Save Best Model
12. Predict Next Word

---

## Model Architecture

- Embedding Layer
- Bidirectional LSTM Layer
- Dropout Layer
- Dense Layer (ReLU)
- Softmax Output Layer

---

## Results

### Accuracy

[View Accuracy Graph](graphs/accuracy.png)

![Accuracy](graphs/accuracy.png)

---

### Loss

[View Loss Graph](graphs/loss.png)

![Loss](graphs/loss.png)

---

### Top Words

[View Top Words Graph](graphs/top_words.png)

![Top Words](graphs/top_words.png)

---

### Word Cloud

[View Word Cloud](graphs/wordcloud.png)

![Word Cloud](graphs/wordcloud.png)

---

## How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Preprocess Dataset

```bash
python preprocess.py
```

### Train Model

```bash
python train.py
```

### Run Prediction

```bash
python predict.py
```

---

## Sample Prediction

```
Input:
Artificial intelligence is

Suggestions:
1. transforming
2. changing
3. becoming
```

---

## Future Improvements

- Transformer-based language models
- GPT architecture
- Beam Search Decoding
- Top-k Sampling
- Top-p Sampling
- Web Application using Flask
- REST API Deployment

---

## Author

**Hafsa Asif**

Computer Science Student

Interested in:
- Artificial Intelligence
- Machine Learning
- Deep Learning
- Natural Language Processing
- Data Analysis

---

⭐ If you found this project helpful, consider giving it a star.
