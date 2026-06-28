# 😊 EmotionSense AI

An AI-powered Emotion Detection System built using **Natural Language Processing (NLP)** and **Machine Learning**. The application analyzes user-entered text and predicts the underlying emotion through an interactive Streamlit interface.

---

## 🚀 Live Demo

Coming Soon...

---

## 📌 Features

- 😊 Detect emotions from user-entered text
- 🤖 Machine Learning based emotion classification
- 📊 Displays confidence score for predictions
- 📈 Probability breakdown for all emotion classes
- 🎨 Modern and responsive Streamlit UI
- ⚡ Fast real-time predictions
- 📱 User-friendly interface

---

## 🎯 Supported Emotions

- 😊 Joy
- 😢 Sadness
- 😡 Anger
- 😨 Fear
- ❤️ Love
- 😲 Surprise

---

## 🛠️ Tech Stack

### Frontend
- Streamlit

### Machine Learning
- Logistic Regression
- CountVectorizer
- Scikit-Learn

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- Joblib

---

## 📂 Project Structure

```
EmotionSense-AI/
│
├── app.py
├── train_model.py
├── model.pkl
├── vectorizer.pkl
├── label_encoder.pkl
├── style.css
├── requirements.txt
├── README.md
└── train.txt
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Ammarqasmi03/EmotionSense-AI.git
```

Move into the project directory

```bash
cd EmotionSense-AI
```

Create a virtual environment

```bash
python -m venv emotion_env
```

Activate the environment

### Windows

```bash
emotion_env\Scripts\activate
```

### Linux / macOS

```bash
source emotion_env/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 🧠 Model Details

| Feature | Value |
|----------|--------|
| Algorithm | Logistic Regression |
| Feature Extraction | CountVectorizer |
| Accuracy | **90%** |
| Dataset | Emotion Dataset |
| Language | English |

---

## 📊 Workflow

```
User Input
      │
      ▼
Text Preprocessing
      │
      ▼
CountVectorizer
      │
      ▼
Logistic Regression
      │
      ▼
Emotion Prediction
      │
      ▼
Probability Breakdown
```


---

## 📈 Future Improvements

- Transformer-based emotion classification (BERT)
- Voice-based emotion detection
- Multilingual support
- Sentiment visualization dashboard
- REST API using FastAPI
- Docker deployment

---

## ⚠️ Limitations

This project uses a traditional Bag-of-Words approach with CountVectorizer and Logistic Regression. While it achieves **90% accuracy**, predictions may be less accurate for sentences that differ significantly from the training data because the model does not capture semantic meaning like transformer-based models.

---

## 👨‍💻 Author

**Ammar**

MCA (Artificial Intelligence & Machine Learning)

Jamia Millia Islamia

---

## ⭐ If you like this project

Please consider giving it a ⭐ on GitHub!
