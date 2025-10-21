import streamlit as st
import joblib
import numpy as np

# =====================================
# 🔧 PAGE CONFIGURATION
# =====================================
st.set_page_config(
    page_title="Heart Disease Predictor 💓",
    page_icon="❤️",
    layout="centered"
)

# =====================================
# 📦 LOAD TRAINED MODEL
# =====================================
model = joblib.load('heart_disease_model.pkl')

# =====================================
# 🖼️ CUSTOM STYLES (Responsive)
# =====================================
st.markdown("""
    <style>
        /* Mobile button fixed at bottom */
        @media (max-width: 768px) {
            .stButton>button {
                position: fixed;
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%);
                width: 90% !important;
                z-index: 9999;
            }

            /* Add bottom padding to content */
            .block-container {
                padding-bottom: 100px !important;
                overflow-y: auto;
            }

            /* Stack inputs vertically */
            .stNumberInput, .stSelectbox {
                width: 100% !important;
            }

            h1 {
                font-size: 1.6rem !important;
            }
        }

        /* Sidebar styling */
        .sidebar-content {
            font-size: 14px;
            line-height: 1.6;
        }

        .model-info-box {
            background: rgba(0, 180, 255, 0.08);
            padding: 10px;
            border-radius: 8px;
            border: 1px solid rgba(0, 180, 255, 0.2);
            margin-top: 10px;
        }

        .disclaimer {
            font-size: 12px;
            color: #999;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# =====================================
# 📝 SIDEBAR - ABOUT SECTION
# =====================================
st.sidebar.title("📊 About This Application")

st.sidebar.markdown("""
<div class="sidebar-content">
<b>Title:</b> Heart Disease Risk Prediction Using Machine Learning  
<b>Objective:</b> Predicts the likelihood of heart disease based on clinical parameters using AI models.

<b>Dataset:</b> UCI Heart Disease Dataset  
<b>Technologies:</b> Python, Streamlit, Scikit-learn 
</div>
""", unsafe_allow_html=True)

# 📈 Model Info Box
st.sidebar.markdown("""
<div class="model-info-box">
<b>🧠 Model Information</b><br>
Algorithm: Logistic Regression<br>
Accuracy: 85% (Test Data)<br>
Evaluation Metrics: Precision, Recall, F1-score
</div>
""", unsafe_allow_html=True)

# ⚠️ Disclaimer
st.sidebar.markdown("""
<div class="disclaimer">
⚠️ <b>Disclaimer:</b><br>
This application is for academic and research purposes only.<br>
It is not a medical diagnostic tool.
</div>
""", unsafe_allow_html=True)

# =====================================
# 🧠 TITLE & INTRO
# =====================================
st.markdown(
    """
    <h1 style='text-align: center; font-size: 2.5rem;'>
        ❤️ Heart Disease Predictor
    </h1>
    <p style='text-align:center; color:gray; font-size: 1rem;'>
        AI-powered heart disease risk assessment based on clinical data
    </p>
    """,
    unsafe_allow_html=True
)

# =====================================
# 🧍 INPUT FORM SECTION
# =====================================
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 50, 250)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0, 1])
    restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])

with col2:
    thalach = st.number_input("Max Heart Rate Achieved", 50, 250)
    exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST (0–2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thal (3 = Normal, 6 = Fixed Defect, 7 = Reversible Defect)", [3, 6, 7])

# =====================================
# 🤖 PREDICTION
# =====================================
if st.button("🔮 Predict Heart Disease"):
    # Prepare input data for model
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    st.subheader("🩺 Prediction Result")
    if prediction[0] == 1:
        st.error("""
        ⚠️ **High Risk Detected**  
        Based on the provided clinical parameters, the model indicates a *higher likelihood of heart disease*.
        """)
    else:
        st.success("""
        ✅ **Low Risk**  
        Based on the provided clinical parameters, the model indicates a *low likelihood of heart disease*.
        """)

# =====================================
# 🪶 FOOTER
# =====================================
st.markdown("""
<hr>
<p style='text-align:center; color:#00f7ff; font-size:14px;'>
Developed with ❤️ by <b>Aniket Lad</b><br>
</p>
""", unsafe_allow_html=True)
