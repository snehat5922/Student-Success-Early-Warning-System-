import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
from utils import calculate_risk_score, recommendation

st.set_page_config(
    page_title="Student Success Analytics",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.metric-card {
    background-color:white;
    padding:15px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

model = joblib.load("models/risk_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")

st.title("🎓 Student Success & Early Warning System")

uploaded_file = st.file_uploader(
    "Upload Student Dataset",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df)

    # KPI Section
    st.subheader("📊 Academic KPIs")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Students", len(df))
    col2.metric("Avg Attendance", f"{df['Attendance'].mean():.1f}%")
    col3.metric("Avg CGPA", f"{df['CGPA'].mean():.2f}")
    col4.metric("Avg Quiz Score", f"{df['Quiz_Score'].mean():.1f}")

    # Attendance Histogram
    st.subheader("Attendance Distribution")

    fig = px.histogram(
        df,
        x="Attendance",
        nbins=20,
        title="Attendance Analysis"
    )

    st.plotly_chart(fig, use_container_width=True)

    # CGPA vs Attendance
    st.subheader("CGPA vs Attendance")

    fig2 = px.scatter(
        df,
        x="Attendance",
        y="CGPA",
        color="Risk",
        size="Quiz_Score",
        hover_data=["Student_ID"]
    )

    st.plotly_chart(fig2, use_container_width=True)

    # LMS Activity
    st.subheader("LMS Engagement")

    fig3 = px.box(
        df,
        y="LMS_Clicks",
        title="Student LMS Activity"
    )

    st.plotly_chart(fig3, use_container_width=True)

    # Risk Prediction

    prediction_data = df.drop(
        ["Student_ID", "Risk"],
        axis=1
    )

    predictions = model.predict(prediction_data)

    df["Predicted Risk"] = encoder.inverse_transform(predictions)

    df["Risk Score"] = df.apply(calculate_risk_score, axis=1)

    df["Recommendation"] = df.apply(
        recommendation,
        axis=1
    )

    st.subheader("Predicted Student Risk")

    st.dataframe(df)

    # Risk Pie Chart

    st.subheader("Risk Distribution")

    pie = px.pie(
        df,
        names="Predicted Risk"
    )

    st.plotly_chart(
        pie,
        use_container_width=True
    )

    # Top Risk Students

    st.subheader("🚨 High Risk Students")

    high_risk = df[
        df["Predicted Risk"] == "High"
    ]

    st.dataframe(high_risk)

    # Recommendation Section

    st.subheader("Intervention Suggestions")

    st.dataframe(
        df[
            [
                "Student_ID",
                "Predicted Risk",
                "Recommendation"
            ]
        ]
    )
