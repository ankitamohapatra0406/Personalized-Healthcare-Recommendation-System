import streamlit as st
import pickle

model = pickle.load(open("models/disease_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))

st.set_page_config(page_title="Healthcare Recommendation System")

st.title("🩺 Personalized Healthcare Recommendation System")
st.write("Enter your symptoms to get a disease prediction and medicine recommendation.")

symptoms_input = st.text_input(
    "Enter symptoms (comma separated)",
    placeholder="fever, cough, headache"
)

if st.button("Get Recommendation"):
    if symptoms_input.strip() == "":
        st.warning("Please enter symptoms.")
    else:
        symptoms_vec = vectorizer.transform([symptoms_input])

        disease_encoded = model.predict(symptoms_vec)
        disease = label_encoder.inverse_transform(disease_encoded)[0]

        st.success(f"Predicted Disease: **{disease}**")

        st.info("Please consult a doctor before taking any medication.")
