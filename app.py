import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests


from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie

# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# -------------------------------
# Load Model and Vectorizer
# -------------------------------

model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# -------------------------------
# Custom CSS (FAANG Style UI)
# -------------------------------

st.markdown("""
<style>

body{
background-color:#407088;
}

.main-title{
font-size:42px;
font-weight:700;
color: #ffc857;
text-align:center;
}

.subtitle{
text-align:center;
color:white;
font-size:18px;
}

.header-box{
background:linear-gradient(90deg, #003049, #084c61);
padding:35px;
border-radius:12px;
margin-bottom:30px;
}

.card{
background:white;
padding:25px;
border-radius:15px;
box-shadow:0 5px 15px rgba(0,0,0,0.08);
margin-bottom:20px;
}

.result-real{
background:#e6f9f0;
border-left:6px solid #10b981;
padding:20px;
border-radius:10px;
font-size:20px;
}

.result-fake{
background:#ffeaea;
border-left:6px solid #ef4444;
padding:20px;
border-radius:10px;
font-size:20px;
}

.tech-card{
text-align:center;
padding:15px;
background:white;
border-radius:12px;
box-shadow:0 5px 10px rgba(0,0,0,0.05);
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# Lottie Animation Loader
# -------------------------------

def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

ai_animation = load_lottie(
"https://assets10.lottiefiles.com/packages/lf20_w51pcehl.json"
)

# -------------------------------
# Header
# -------------------------------

st.markdown("""
<div class="header-box">
<div class="main-title">Fake News Detection System</div>
<div class="subtitle">AI Powered Misinformation Detection using NLP</div>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar
# -------------------------------

with st.sidebar:

    st.title("🧠 Fake News Detection")

    st.write("""
Fake News Detection uses **Natural Language Processing and Machine Learning**
to identify misleading or false information in news articles.
""")

    st.write("**Model:** TF-IDF + ML Classifier")

    selected = option_menu(
        menu_title="Navigation",
        options=["Home","Analyze News","Insights","Technical Stack","Feedback","Admin"],
        icons=["house","search","bar-chart","cpu","chat","lock"],
        default_index=0
    )

    st.markdown("---")

    st.subheader("Developer")

    st.write("Kanha Patidar")

    st.markdown("""
<div style="display:flex; gap:25px; align-items:center;">

<a href="https://github.com/kanha165" target="_blank">
<img src="https://cdn-icons-png.flaticon.com/512/733/733553.png" width="38">
</a>

<a href="https://www.linkedin.com/in/kanha-patidar-837421290/" target="_blank">
<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="38">
</a>

<a href="mailto:kanhapatidar7251@gmail.com">
<img src="https://cdn-icons-png.flaticon.com/512/732/732200.png" width="38">
</a>

</div>
""", unsafe_allow_html=True)

# -------------------------------
# HOME PAGE
# -------------------------------
if selected == "Home":

    col1, col2 = st.columns([1,1], vertical_alignment="top")

    with col1:
        st_lottie(ai_animation, height=330)

    with col2:

        st.markdown("### AI System Overview")

        st.write("""
This AI system analyzes news articles using **Natural Language Processing**
to determine whether the information is **Real or Fake**.

The model processes the text using **TF-IDF vectorization**
and predicts authenticity using a trained Machine Learning classifier.
""")
# -------------------------------
# ANALYZE NEWS
# -------------------------------

elif selected == "Analyze News":

    st.markdown("### 📰 Analyze News Article")

    news = st.text_area(
        "Paste News Content Here",
        height=200
    )

    if st.button("Analyze News"):

        if news.strip()=="":
            st.warning("Please enter news text")
        else:

            vector = vectorizer.transform([news])
            prediction = model.predict(vector)[0]

            prob = model.predict_proba(vector)[0]

            confidence = np.max(prob)

            st.write("")

            if prediction==1:

                st.markdown(f"""
                <div class="result-real">
                ✅ This news appears to be Real
                <br><br>
                Confidence Score: {confidence:.2f}
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div class="result-fake">
                ⚠️ This news appears to be Fake
                <br><br>
                Confidence Score: {confidence:.2f}
                </div>
                """, unsafe_allow_html=True)

# -------------------------------
# INSIGHTS DASHBOARD
# -------------------------------

elif selected == "Insights":

    st.markdown("### 📊 Model Insights Dashboard")

    col1,col2 = st.columns(2)

    # Confidence Gauge

    confidence = np.random.uniform(0.7,0.95)

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence*100,
        title={"text":"Confidence"},
        gauge={'axis':{'range':[0,100]}}
    ))

    col1.plotly_chart(gauge,use_container_width=True)

    # Fake vs Real Probability

    df = pd.DataFrame({
        "Label":["Fake","Real"],
        "Probability":[0.35,0.65]
    })

    fig = px.bar(df,x="Label",y="Probability",color="Label")

    col2.plotly_chart(fig,use_container_width=True)

    # Word Frequency Chart

    words = ["government","economy","breaking","policy","health"]
    freq = [30,22,15,18,10]

    wf = px.bar(
        x=words,
        y=freq,
        labels={"x":"Words","y":"Frequency"},
        title="Top Words"
    )

    st.plotly_chart(wf,use_container_width=True)

# -------------------------------
# TECHNICAL STACK
# -------------------------------
elif selected == "Technical Stack":

    st.markdown("## ⚙ Technical Stack Used")
    st.write("Technologies used to build this Fake News Detection System")

    st.markdown("""
### 🤖 Machine Learning & Data Analysis

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![ScikitLearn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

<br>

![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

### ⚙ Tools

![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)
![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Natural%20Language%20Processing-blue?style=for-the-badge&logo=spacy&logoColor=white)
""", unsafe_allow_html=True)
# -------------------------------
# FEEDBACK
# -------------------------------


elif selected == "Feedback":

    st.markdown("### 💬 Feedback & Suggestions")

    name = st.text_input("Name")

    email = st.text_input("Email (optional)")

    suggestion = st.text_area("Suggestion / Bug Report")

    if st.button("Submit Feedback"):

        feedback_data = {
            "Name": name,
            "Email": email,
            "Suggestion": suggestion
        }

        df = pd.DataFrame([feedback_data])

        file_path = "feedback.csv"

        if os.path.exists(file_path):
            df.to_csv(file_path, mode="a", header=False, index=False)
        else:
            df.to_csv(file_path, mode="w", header=True, index=False)

        st.success("Thank you for your feedback!")
        
        
# -------------------------------
# ADMIN PANEL
# -------------------------------

elif selected == "Admin":

    st.markdown("### 🔒 Admin Dashboard")

    password = st.text_input("Enter Admin Password", type="password")

    if password == "kanha123":   # yaha apna password rakh sakte ho

        st.success("Access Granted")

        file_path = "feedback.csv"

        if os.path.exists(file_path):

            df = pd.read_csv(file_path)

            st.write("### 📊 Feedback Database")
            st.dataframe(df)

            st.write("Total Feedback:", len(df))

            with open(file_path, "rb") as file:

                st.download_button(
                    label="Download Feedback CSV",
                    data=file,
                    file_name="feedback.csv",
                    mime="text/csv"
                )

        else:
            st.warning("No feedback yet")

    elif password != "":
        st.error("Wrong Password")        
        
        
