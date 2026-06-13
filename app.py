import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("🎓 AI-Powered Student Success & Risk Analysis System")
st.markdown("""
This AI system predicts student success probability,
analyzes academic performance, and identifies risk levels
using Machine Learning.
""")
st.info("Model Accuracy: 98.50%")

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

lunch = st.selectbox(
    "Lunch Type",
    ["Standard", "Free/Reduced"]
)

test_prep = st.selectbox(
    "Test Preparation",
    ["None", "Completed"]
)

math_score = st.slider(
    "Math Score",
    0,
    100,
    60
)

reading_score = st.slider(
    "Reading Score",
    0,
    100,
    60
)

writing_score = st.slider(
    "Writing Score",
    0,
    100,
    60
)

if st.button("Predict Success"):

    gender_val = 1 if gender == "Male" else 0
    lunch_val = 1 if lunch == "Standard" else 0
    test_val = 0 if test_prep == "None" else 1

    data = np.array([
        [
            gender_val,
            lunch_val,
            test_val,
            math_score,
            reading_score,
            writing_score
        ]
    ])

    prediction = model.predict(data)
    probability = model.predict_proba(data)[0][1] * 100

    average = (
        math_score +
        reading_score +
        writing_score
    ) / 3

    st.metric("Average Score", f"{average:.1f}")
    st.metric("Success Probability", f"{probability:.2f}%")
    if probability >= 80:
        st.success("🟢 Risk Level: LOW")

    elif probability >= 50:
        st.warning("🟡 Risk Level: MEDIUM")

    else:
        st.error("🔴 Risk Level: HIGH")

    if prediction[0] == 1:

        st.success(
            "✅ Student has HIGH probability of success."
        )

        st.balloons()

    else:

        st.error(
            "⚠️ Student may need improvement."
        )

        st.write("Suggestions:")

        st.write("- Improve study consistency")
        st.write("- Complete preparation courses")
        st.write("- Focus on weak subjects")