import pandas as pd
import streamlit as st
import joblib
model = joblib.load("exercise_model.pkl")
le = joblib.load("label_encoder.pkl")
df = pd.read_csv("final_exercise_dataset.csv")
body_part = sorted(df["bodyPart"].dropna().unique())
equipmentt = sorted(df["equipment"].dropna().unique())
targett = sorted(df["target"].dropna().unique())
secondary_muscle = sorted(df["secondaryMuscles"].dropna().unique())
categoryy = sorted(df["category"].dropna().unique())

st.title("Exercise Difficulty Predictor")
st.subheader("Using Machine Learning")
st.write("An interactive Machine Learning web app that predicts exercise difficulty levels using exercise-related features and preprocessing pipelines.")
st.caption("built using python, scikit learn and streamlit.")
with st.expander("Dataset Details"):
     st.write("Dataset Name: Exercise Dataset")
     st.write(f"Total Rows: {df.shape[0]}")
     st.write(f"total Columns: {df.shape[1]}")

     st.subheader("Features Used")
     st.write("""
       1. BodyPart
       2. Equipment
       3. Target
       4. Secondary Muscles
       5. Category
       """)
     st.subheader("Target Column")
     st.write("Difficulty")

with st.expander("Model Details"):
     st.write("Model Used: Logistic Regression")
     st.subheader("Preprocessing Details")
     st.write("""
       1. Handled missing values using SimpleImputer
       2. Encode the categorical column using OneHotEncoder
       3. Encode the target column using Label Encoder
       4. Pipeline And ColumnTransformer used
       """)
     st.subheader("Model Accuracy")
     st.write("Accuracy : 1")

with st.expander("Prediction"):
       bodyPart = st.selectbox("Select BODY PART", body_part)
       equipment = st.selectbox("Select EQUIPMENT", equipmentt)
       target = st.selectbox("Select TARGET", targett)
       secondaryMuscles = st.selectbox("Select SECONDARY MUSCLE", secondary_muscle)
       category = st.selectbox("Select CATEGORY", categoryy)

       submit = st.button("Predict")
       if submit:
              sample = {
                     'bodyPart' : [bodyPart],
                     'target' : [target],
                     'category': [category],
                     'secondaryMuscles' : [secondaryMuscles],
                     'equipment' :  [equipment]
              }
              sample_df = pd.DataFrame(sample)
              prediction = model.predict(sample_df)
              result = le.inverse_transform(prediction)
              st.success(f"Predicted Difficulty: {result[0]}")

