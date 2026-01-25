import pandas as pd
import pickle

df = pd.read_csv("data/medical_data_cleaned.csv")

model = pickle.load(open("models/disease_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))

def recommend_medicine(symptoms_input):
    symptoms_vec = vectorizer.transform([symptoms_input])

    # Predict disease
    disease_encoded = model.predict(symptoms_vec)
    disease = label_encoder.inverse_transform(disease_encoded)[0]

    # Get medicine from dataset
    medicine = df[df['Disease'] == disease]['Medicine'].iloc[0]

    return disease, medicine


if __name__ == "__main__":
    test_symptoms = "fever cough"
    disease, medicine = recommend_medicine(test_symptoms)
    print("Predicted Disease:", disease)
    print("Recommended Medicine:", medicine)
