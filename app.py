import streamlit as st
import pandas as pd
import joblib
from keras.models import load_model


# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide"
)


# =====================================
# LOAD MODEL
# =====================================

@st.cache_resource
def load_assets():
    model = load_model("heart_disease.keras")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor


model, preprocessor = load_assets()


# =====================================
# HEADER
# =====================================

st.title("❤️ Heart Disease Risk Predictor")

st.markdown(
    "### AI-powered cardiovascular risk assessment"
)

st.info(
    "Enter the patient's health information below to "
    "generate a machine-learning prediction."
)


# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("About")

st.sidebar.write(
    """
    This application uses a Deep Learning neural
    network to predict the likelihood of heart disease
    based on patient health and lifestyle information.
    """
)

st.sidebar.divider()

st.sidebar.write("**Model:** Neural Network")
st.sidebar.write("**Framework:** TensorFlow / Keras")
st.sidebar.write("**UI:** Streamlit")


# =====================================
# PATIENT INFORMATION
# =====================================

st.header("👤 Patient Information")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50
    )

    sex = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )

    systolic = st.number_input(
        "Resting BP Systolic",
        min_value=50,
        max_value=250,
        value=120
    )

    diastolic = st.number_input(
        "Resting BP Diastolic",
        min_value=30,
        max_value=150,
        value=80
    )

    cholesterol = st.number_input(
        "Total Cholesterol",
        min_value=50,
        max_value=600,
        value=200
    )

    hdl = st.number_input(
        "HDL",
        min_value=10,
        max_value=150,
        value=50
    )

    ldl = st.number_input(
        "LDL",
        min_value=10,
        max_value=500,
        value=100
    )


with col2:

    triglycerides = st.number_input(
        "Triglycerides",
        min_value=20,
        max_value=1000,
        value=150
    )

    fasting = st.number_input(
        "Fasting Blood Sugar",
        min_value=50,
        max_value=400,
        value=100
    )

    hba1c = st.number_input(
        "HbA1c",
        min_value=3.0,
        max_value=20.0,
        value=5.5
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=70.0,
        value=25.0
    )

    resting_hr = st.number_input(
        "Resting Heart Rate",
        min_value=30,
        max_value=220,
        value=70
    )

    max_hr = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        [
            "Typical Angina",
            "Atypical Angina",
            "Non-anginal Pain",
            "Asymptomatic"
        ]
    )


with col3:

    exercise_angina = st.selectbox(
        "Exercise Induced Angina",
        [0, 1],
        format_func=lambda x: "Yes" if x else "No"
    )

    st_depression = st.number_input(
        "ST Depression",
        min_value=0.0,
        max_value=10.0,
        value=1.0
    )

    family_history = st.selectbox(
        "Family History",
        [0, 1],
        format_func=lambda x: "Yes" if x else "No"
    )

    smoker = st.selectbox(
        "Smoker Status",
        ["Never", "Former", "Current"]
    )

    alcohol = st.number_input(
        "Alcohol Units / Week",
        min_value=0.0,
        max_value=100.0,
        value=5.0
    )

    exercise = st.number_input(
        "Exercise Minutes / Week",
        min_value=0.0,
        max_value=2000.0,
        value=150.0
    )

    sleep = st.number_input(
        "Sleep Hours",
        min_value=1.0,
        max_value=15.0,
        value=7.0
    )


# =====================================
# LIFESTYLE
# =====================================

st.header("🏃 Lifestyle Information")

c1, c2, c3 = st.columns(3)

with c1:

    stress = st.slider(
        "Stress Score",
        0.0,
        10.0,
        5.0
    )

with c2:

    steps = st.number_input(
        "Daily Steps",
        min_value=0,
        max_value=50000,
        value=5000
    )

with c3:

    diet = st.slider(
        "Diet Quality Score",
        0.0,
        10.0,
        5.0
    )


wearable = st.selectbox(
    "Wearable Device Owner",
    [0, 1],
    format_func=lambda x: "Yes" if x else "No"
)


# =====================================
# PREDICTION
# =====================================

st.divider()

if st.button(
    "🔍 Predict Heart Disease Risk",
    use_container_width=True
):

    patient = pd.DataFrame([{

        "age": age,
        "sex": sex,

        "resting_bp_systolic": systolic,
        "resting_bp_diastolic": diastolic,

        "cholesterol_total": cholesterol,
        "hdl": hdl,
        "ldl": ldl,
        "triglycerides": triglycerides,

        "fasting_blood_sugar": fasting,
        "hba1c": hba1c,

        "bmi": bmi,
        "resting_heart_rate": resting_hr,
        "max_heart_rate_achieved": max_hr,

        "chest_pain_type": chest_pain,

        "exercise_induced_angina":
            exercise_angina,

        "st_depression":
            st_depression,

        "family_history":
            family_history,

        "smoker_status":
            smoker,

        "alcohol_units_per_week":
            alcohol,

        "exercise_minutes_per_week":
            exercise,

        "sleep_hours":
            sleep,

        "stress_score":
            stress,

        "wearable_owner":
            wearable,

        "daily_steps":
            steps,

        "diet_quality_score":
            diet
    }])


    # Preprocess
    processed = preprocessor.transform(patient)

    if hasattr(processed, "toarray"):
        processed = processed.toarray()


    # Prediction
    probability = model.predict(
        processed,
        verbose=0
    )[0][0]


    risk = probability * 100


    # =================================
    # RESULT
    # =================================

    st.header("📊 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if probability >= 0.5:

            st.error(
                "⚠️ Higher Heart Disease Risk"
            )

        else:

            st.success(
                "✅ Lower Heart Disease Risk"
            )


    with result_col2:

        st.metric(
            "Predicted Probability",
            f"{risk:.2f}%"
        )


    st.progress(
        min(float(probability), 1.0)
    )


    if probability >= 0.5:

        st.warning(
            "The model predicts a higher likelihood "
            "of heart disease. Please consult a qualified "
            "healthcare professional for proper evaluation."
        )

    else:

        st.success(
            "The model predicts a lower likelihood "
            "of heart disease based on the provided data."
        )


# =====================================
# DISCLAIMER
# =====================================

st.divider()

st.caption(
    "⚠️ This application is for educational and "
    "demonstration purposes only. It is not a medical "
    "diagnosis or a substitute for professional medical advice."
)