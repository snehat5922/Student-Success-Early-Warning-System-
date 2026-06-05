import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(
    "data/student_data.csv"
)

X = df.drop(
    ["Student_ID", "Risk"],
    axis=1
)

y = df["Risk"]

encoder = LabelEncoder()

y = encoder.fit_transform(y)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X, y)

joblib.dump(
    model,
    "models/risk_model.pkl"
)

joblib.dump(
    encoder,
    "models/label_encoder.pkl"
)

print("Model Saved Successfully")
