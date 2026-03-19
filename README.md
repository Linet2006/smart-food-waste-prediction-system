# smart-food-waste-prediction-system
ML-powered web app that predicts daily meal demand to reduce food waste in large-scale dining facilities. Built with Random Forest, Flask, and a glassmorphism UI. | LearnDepth ML Internship 2026
# 🍽️ Smart Food Waste Prediction System

A machine learning-powered web application that predicts the optimal number of meals to prepare daily, helping food service environments reduce waste and improve efficiency.

Built as part of the Machine Learning Internship Program at **LearnDepth Academy LLP** (March 2026).

---

## 🚀 Overview

Food waste is a major challenge in large-scale dining environments — restaurants, university cafeterias, hotels, and corporate dining facilities. Traditional planning relies on rough manual estimates, leading to either excess food waste or unmet demand.

This system solves that by training a **Random Forest Regression model** on historical consumption data to deliver accurate, data-driven daily meal predictions through an intuitive Flask web application.

---

## 🧠 Features

- 📊 End-to-end ML pipeline (preprocessing → EDA → feature engineering → training → evaluation)
- 🌲 Random Forest Regressor selected as best-performing model
- 🌐 Flask web application with real-time prediction
- 🎨 Dark glassmorphism UI for a modern user experience
- 💾 Model serialized with Joblib for seamless deployment
- 📈 Evaluated using MAE and RMSE metrics

---


## 🔢 Input Features

| Feature                   | Description                              |
|---------------------------|------------------------------------------|
| Day of Week               | Monday – Sunday                          |
| Weather                   | Sunny / Cloudy / Rainy / Stormy          |
| Festival                  | 1 = Festival day, 0 = Normal day         |
| Expected Customers        | Estimated number of diners               |
| Previous Day Consumption  | Meals consumed the day before            |
| Previous Week Same Day    | Meals consumed on the same day last week |

**Target Variable:** `Meals_Consumed` — predicted number of meals to prepare.

---

## ⚙️ Tech Stack

- **Language:** Python
- **ML Libraries:** Scikit-learn, Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Web Framework:** Flask
- **Model Serialization:** Joblib

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| MAE    | 24.10 |
| RMSE   | 29.61 |



---

## 🛠️ How to Run
```bash
# Clone the repository
git clone https://github.com/your-username/smart-food-waste-prediction-system.git
cd smart-food-waste-prediction-system

# Install dependencies
pip install -r requirements.txt

#tain model
python train_model.py

# Run the Flask app
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

---

## 🎯 Real-World Applications

- University and college cafeterias
- Hotel and restaurant chains
- Corporate dining facilities
- Catering companies
- Large-scale food service organizations

---

## 🏫 Internship Context

This project was developed as part of the **LearnDepth Academy LLP Machine Learning Internship Program (2026)**, focusing on building and deploying end-to-end ML solutions for real-world problems.

--
