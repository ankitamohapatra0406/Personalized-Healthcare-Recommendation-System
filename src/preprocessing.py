import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("data/medical_data_cleaned.csv")

X = df['Symptoms']      # input
y = df['Disease']       # output

# Convert text symptoms to numbers
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Encode target labels (Disease)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y_encoded,
    test_size=0.2,
    random_state=42
)

print("Preprocessing completed successfully")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
