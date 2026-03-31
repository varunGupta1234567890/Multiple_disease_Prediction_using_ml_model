## Health Assistant (ML Web App)
A multi-disease prediction web application built using Streamlit that uses Machine Learning models to predict:
1) Diabetes
2) Heart Disease
3) Parkinson’s Disease
4) Features->
# Simple and interactive UI
# Multiple disease prediction in one app
# Real-time predictions
# Clean layout with sidebar navigation
# Uses trained ML models (.sav files)

## Tech Stack
Python 
Streamlit
Scikit-learn
Pickle

## Project Structure

Health-Assistant/
│
├── saved_models1/
│   ├── diabetes_model.sav
│   ├── heart_disease_model.sav
│   ├── parkinson_model.sav
│
├── app.py
├── requirements.txt
└── README.md

# Installation & Setup
1) Clone the Repository
Bash
git clone https://github.com/your-username/health-assistant.git
cd health-assistant
2) Install Dependencies
Bash
pip install -r requirements.txt
3) Run the App
streamlit run app.py

# How It Works
User inputs medical data
Data is passed to pre-trained ML models
Model predicts whether disease is present or not
Result is displayed instantly