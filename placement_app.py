import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("placement_model.pkl")

st.set_page_config(page_title="Smart Placement Prediction", page_icon="🎓")
st.title("🎓 Smart Placement Prediction System")
st.markdown("Get a personalized prediction on your chances of being placed along with improvement tips!")

# Student Input Form
with st.form("placement_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    stream = st.selectbox("Stream", ["CSE", "IT", "ECE", "EEE", "MECH", "CIVIL"])
    tenth = st.slider("10th Percentage", 40, 100, 75)
    twelfth = st.slider("12th Percentage", 40, 100, 75)
    cgpa = st.slider("Current CGPA", 5.0, 10.0, 7.0)
    aptitude = st.slider("Aptitude Score (out of 100)", 0, 100, 60)
    technical = st.slider("Technical Skill Score (out of 100)", 0, 100, 60)
    internships = st.slider("Internships Completed", 0, 5, 1)
    projects = st.slider("Major Projects Done", 0, 5, 1)
    submit = st.form_submit_button("Predict")

# Encode categorical variables
stream_dict = {"CSE": 0, "CIVIL": 1, "ECE": 2, "EEE": 3, "IT": 4, "MECH": 5}
gender_val = 1 if gender == "Male" else 0
stream_val = stream_dict[stream]

if submit:
    user_data = pd.DataFrame([[
        gender_val, stream_val, tenth, twelfth, cgpa,
        aptitude, technical, internships, projects
    ]], columns=["gender", "stream", "tenth_score", "twelfth_score", "cgpa",
                 "aptitude_score", "technical_score", "internships", "projects"])

    prediction = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0][1]

    if prediction == 1:
        st.success(f"✅ You are likely to be placed! 🎉 Probability: {probability*100:.2f}%")
    else:
        st.error(f"❌ You might not be placed. Probability: {probability*100:.2f}%")

    st.markdown("### 📘 Suggestions for Improvement:")
    if cgpa < 7.0:
        st.write("- Try to increase your CGPA above 7.0")
    if aptitude < 60:
        st.write("- Practice aptitude tests regularly (Quant, Reasoning, etc.)")
    if technical < 60:
        st.write("- Improve technical skills through coding practice and mini-projects")
    if internships == 0:
        st.write("- Gain internship experience to boost your profile")
    if projects < 2:
        st.write("- Work on more academic/personal projects")

    st.info("📢 This prediction is AI-based and for guidance only.")
