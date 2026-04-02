import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕")

st.markdown("""
<style>
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

working_dir = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    with open(os.path.join(working_dir, "saved_models1", "diabetes_model.sav"), "rb") as f:
        diabetes = pickle.load(f)

    with open(os.path.join(working_dir, "saved_models1", "heart_disease_model.sav"), "rb") as f:
        heart = pickle.load(f)

    with open(os.path.join(working_dir, "saved_models1", "parkinson_model.sav"), "rb") as f:
        park = pickle.load(f)

    return diabetes, heart, park

diabetes_model, heart_model, parkinson_model = load_models()

with st.sidebar:
    selected = option_menu(
        '🧑‍⚕ Health Assistant',
        ['Diabetes Prediction',
         'Heart Disease Prediction',
         'parkinson Prediction'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

st.sidebar.success("Select a prediction type")

# ================== DIABETES Prediction ==================
if selected == 'Diabetes Prediction':

    st.title('🩺 Diabetes Prediction')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input("Pregnancies", min_value=0, value=2)
        SkinThickness = st.number_input("Skin Thickness", value=20)
        DPF = st.number_input("Pedigree Function", value=0.5)

    with col2:
        Glucose = st.number_input("Glucose", value=120)
        Insulin = st.number_input("Insulin", value=80)
        Age = st.number_input("Age", min_value=1, value=45)

    with col3:
        BP = st.number_input("Blood Pressure", value=70)
        BMI = st.number_input("BMI", value=25.5)

    if st.button("🔍 Predict Diabetes"):
        inputs = [Pregnancies, Glucose, BP, SkinThickness,
                  Insulin, BMI, DPF, Age]

        result = diabetes_model.predict([inputs])

        if result[0] == 1:
            st.error("⚠️ Diabetic")
        else:
            st.success(" Not Diabetic")

# ================== HEART Disease Prediction ==================
if selected == 'Heart Disease Prediction':

    st.title('❤️ Heart Disease Prediction')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        trestbps = st.number_input("Resting Blood Pressure", value=120)
        restecg = st.selectbox("Rest ECG", [0, 1, 2])
        oldpeak = st.number_input("Oldpeak", value=1.2)
        thal = st.selectbox("Thal", [0, 1, 2])

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        sex = 1 if gender == "Male" else 0

        chol = st.number_input("Cholesterol", value=200)
        thalach = st.number_input("Max Heart Rate", value=150)
        slope = st.selectbox("Slope", [0, 1, 2])

    with col3:
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        fbs = st.selectbox("Fasting Blood Sugar >120", [0, 1])
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        ca = st.selectbox("Number of Major Vessels (CA)", [0, 1, 2, 3])

    if st.button("🔍 Predict Heart Disease"):

        inputs = [age, sex, cp, trestbps, chol, fbs,
                  restecg, thalach, exang, oldpeak,
                  slope, ca, thal]

        result = heart_model.predict([inputs])

        if result[0] == 1:
            st.error("⚠️ Heart Disease Detected")
        else:
            st.success(" Healthy Heart")

# ================== PARKINSON Disease Prediction ==================
if selected == "parkinson Prediction":

    st.title("🧠 Parkinson's Disease Prediction")

    features_info = {
        "Fo": "Voice ka base frequency (Hz)",
        "Fhi": "Highest frequency of voice",
        "Flo": "Lowest frequency of voice",
        "Jitter %": "Voice frequency variation (%)",
        "Jitter Abs": "Absolute frequency variation",
        "RAP": "Short-term frequency variation",
        "PPQ": "Pitch variation measure",
        "DDP": "Derived jitter value",
        "Shimmer": "Loudness variation",
        "Shimmer dB": "Loudness variation in decibels",
        "APQ3": "Short-term amplitude variation",
        "APQ5": "Amplitude variation (5 samples)",
        "APQ": "Overall amplitude variation",
        "DDA": "Shimmer related variation",
        "NHR": "Noise-to-Harmonics ratio",
        "HNR": "Harmonics-to-Noise ratio",
        "RPDE": "Signal randomness",
        "DFA": "Signal complexity",
        "Spread1": "Frequency variation pattern",
        "Spread2": "Frequency spread",
        "D2": "Signal complexity (non-linear)",
        "PPE": "Pitch variation entropy"
    }

    inputs = []

    labels = list(features_info.keys())

    cols = st.columns(4)

    for i, label in enumerate(labels):
        with cols[i % 4]:
            val = st.number_input(
                f"{label} ℹ️",
                value=0.0,
                help=features_info[label]
            )
            inputs.append(val)

    with st.expander("ℹ️ Understand Features (Click to expand)"):
        for k, v in features_info.items():
            st.write(f"**{k}**: {v}")

    if st.button("🔍 Predict Parkinson's"):

        result = parkinson_model.predict([inputs])

        if result[0] == 1:
            st.error("⚠️ Parkinson's Detected")
        else:
            st.success(" No Parkinson's")