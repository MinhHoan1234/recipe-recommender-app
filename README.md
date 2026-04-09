# 🍔 Smart AI Recipe Recommender System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([CHÈN_LINK_STREAMLIT_CỦA_BẠN_VÀO_ĐÂY])

## 📌 Project Overview
This is an End-to-End Machine Learning project that recommends cooking recipes based on the **Food.com dataset**. The system acts as a smart kitchen assistant, helping users discover new meals either by exploring what ingredients they currently have in their fridge or by analyzing their past rating behaviors.

## ✨ Key Features
The application provides two core recommendation engines:
* **🥦 Fridge Search Engine (Content-Based Filtering):** Users can select ingredients they currently have (e.g., chicken, garlic, tomatoes) from a dynamically generated vocabulary list. The AI uses **TF-IDF Vectorization** and **Cosine Similarity** to suggest recipes that best match the selected ingredients.
* **👥 Personalized Recommendations (Collaborative Filtering):** Returning users can enter their unique User ID. The system utilizes **Singular Value Decomposition (SVD)** to predict the star rating a user would give to unseen recipes, recommending the Top 5 highest-predicted meals.

## 📁 Project Structure

The repository is organized following modular ML engineering best practices:

```text
.
├── data/                   # Contains raw and processed dataset (Ignored in Git)
├── models/                 # Serialized Machine Learning artifacts (.pkl)
│   ├── df_metadata.pkl
│   ├── recommender_model_v1.pkl
│   ├── tfidf_matrix.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/              # Jupyter notebooks for EDA, model training, and evaluation
├── src/                    # Source code for modularized backend logic
│   ├── __init__.py
│   ├── data_preprocessing.py # Data cleaning and long-tail pruning scripts
│   └── inference.py          # Core AI prediction and logic functions
├── .gitignore
├── app.py                  # Streamlit frontend application
├── README.md
└── requirements.txt        # Project dependencies

```
## 🛠️ Technology Stack
Language: Python 3.12

Data Processing & EDA: Pandas, Numpy, Plotly

Machine Learning: Scikit-learn (TF-IDF, Cosine Similarity), Scikit-Surprise (SVD, KNN)

Web Framework & Deployment: Streamlit, Streamlit Community Cloud

## 🚀 How to Run Locally
If you want to run this project on your local machine, follow these steps:

1. Clone the repository

```Bash
git clone [https://github.com/MinhHoan1234/recipe-recommender-app.git](https://github.com/MinhHoan1234/recipe-recommender-app.git)
cd recipe-recommender-app
```
2. Install dependencies

```Bash
pip install -r requirements.txt
```

3. Run the Streamlit App

```Bash
streamlit run app.py
```

## 🧠 Technical Highlights & Challenges Solved
Overcoming Memory Constraints: Handled a MemoryError during the Collaborative Filtering phase by implementing a "Pruning the Long Tail" strategy, filtering out inactive users and niche recipes to optimize the interaction matrix.

Addressing OOV (Out-Of-Vocabulary): Upgraded the frontend UI with a multiselect dropdown extracted directly from the TF-IDF feature names, ensuring zero spelling errors from user inputs.

Modular Architecture: Decoupled the frontend (app.py) from the complex backend ML logic (src/inference.py) for better scalability and maintainability.