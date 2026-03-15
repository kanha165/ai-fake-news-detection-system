# 📰 AI Fake News Detection System

An AI-powered **Fake News Detection System** built using **Natural Language Processing (NLP), TF-IDF, and Machine Learning**.
The system analyzes news text and predicts whether the news is **Fake or Real** using trained machine learning models and an interactive **Streamlit web application**.

---

# 🚀 Features

* 🧠 NLP based text preprocessing
* 📑 TF-IDF feature engineering (20,000 features + n-grams)
* 🤖 Machine Learning models (Logistic Regression & Linear SVM)
* 📊 Model evaluation using Confusion Matrix & ROC Curve
* ⚡ Interactive Fake News Detection Web App using Streamlit
* 💬 Feedback system to collect user suggestions
* 📈 Cross-Validation for model stability analysis

---

# 🛠 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* Natural Language Processing (NLP)
* Streamlit
* Matplotlib
* Seaborn

---

# 📂 Project Structure

```
fake-news-detection-system
│
├── app.py
│
├── model
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── data
│   └── fake_news.csv
│
├── notebook
│   └── model_train.ipynb
│
├── visuals
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── feedback.csv
├── requirements.txt
└── README.md
```

---

# ▶️ Run Locally

Clone the repository

```
git clone https://github.com/kanha165/fake-news-detection-system.git
```

Go to the project directory

```
cd fake-news-detection-system
```

Install dependencies

```
pip install -r requirements.txt
```

Run the Streamlit application

```
streamlit run app.py
```

---

# 🌐 Live Deployment

This project can be deployed using **Streamlit Cloud or Render**.

Start command:

```
streamlit run app.py --server.port $PORT
```

---

# 📊 Dataset Information

Dataset used: **WELFake Dataset**

* Total News Articles: **71,537**
* Real News: **36,509**
* Fake News: **35,028**

The dataset is balanced which helps improve model performance and reduce bias.

---

# 🤖 Model Performance

Best Model: **Logistic Regression**

| Metric | Score |
|------|------|
| Train Accuracy | **96.93%** |
| Test Accuracy | **95.49%** |
| Cross Validation Accuracy | **95.70%** |

The model demonstrates stable performance with minimal overfitting and good generalization on unseen data.
---

# 🧠 Machine Learning Pipeline

```
Dataset
↓
Text Cleaning
↓
Stopwords Removal
↓
TF-IDF Vectorization
↓
Model Training
↓
Model Evaluation
↓
Streamlit Web Application
```

---

# 📊 Visualizations

The project includes several model evaluation visualizations:

* Confusion Matrix
* ROC Curve
* Dataset Distribution (Fake vs Real)

These visualizations help analyze model performance.

---

# 📌 Project Purpose

The purpose of this project is to demonstrate how **Artificial Intelligence and Natural Language Processing** can help detect misinformation and fake news in digital media.

This system can be used to:

* Detect misleading or fake news articles
* Assist in content verification
* Reduce misinformation spread on social media

---

# 👨‍💻 Author

**Kanha Patidar**
B.Tech CSIT Student
Chamelidevi Group of Institutions, Indore

GitHub:
https://github.com/kanha165
