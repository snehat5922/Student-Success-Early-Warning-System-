import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

model = joblib.load(
    "models/risk_model.pkl"
)

df = pd.read_csv(
    "data/student_data.csv"
)

X = df.drop(
    ["Student_ID", "Risk"],
    axis=1
)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

shap.summary_plot(
    shap_values,
    X
)

plt.show()
