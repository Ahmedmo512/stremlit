import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time as time

st.set_page_config(page_title="Stroke Prediction Dashboard", layout="wide")
st.title('🧠 Stroke Prediction & Visualization Dashboard')
st.image("https://my.clevelandclinic.org/-/scassets/images/org/patient-experience/patient-stories/173-advanced-stroke-procedure-saves-patient-after-deep-brain-bleed/deep-brain-bleeds-new-2.gif", caption="Real-time visualization of a deep brain stroke — emphasizing the urgency of early detection and prevention.", use_container_width=True)

# Sidebar navigation
menu = st.sidebar.radio("Navigate", ["📋 Predict", "📊 Visualizations"])

# Load data
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

data_location = "df_cleaned.csv"
data = load_data(data_location)
data.drop(['work_type', 'Residence_type'], axis=1, inplace=True)

# Split data
X = data.drop('stroke', axis=1)
y = data['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------- Predict Tab -------------------------
if menu == "📋 Predict":

    with st.form(key='prediction_form'):
        st.subheader('Enter Patient Details')
        col1, col2 = st.columns(2)
        with col1:
            sex = st.selectbox('Gender', ['male', 'female'])
            ever_married = st.selectbox('Ever Married', ['Yes', 'No'])
            smoking_status = st.selectbox('Smoking Status', ['Yes', 'Never smoke'])
        with col2:
            hypertension = st.selectbox('Hypertension', ['Ever had hypertension', 'Not Hade hypertension'])
            heart_disease = st.selectbox('Heart Disease', ['Ever had heart_disease', 'Not Hade heart_disease'])

        col6, col7, col8 = st.columns(3)
        with col6:
            age = st.slider("Age", 1, 100, 50)
        with col7:
            bmi = st.slider("BMI", 10, 35, 20)
        with col8:
            avg_glucose_level = st.slider("Avg Glucose Level", 50, 270, 150)

        submit_button = st.form_submit_button(label='Confirm')

    df = pd.DataFrame({
        'sex': [sex],
        'age': [age],
        'hypertension': [1 if hypertension == 'Ever had hypertension' else 0],
        'heart_disease': [1 if heart_disease == 'Ever had heart_disease' else 0],
        'ever_married': [1 if ever_married == 'Yes' else 0],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [0 if smoking_status == 'Never smoke' else 1]
    })

    if submit_button:
        result = model.predict(df)[0]
        neg_perc, pos_perc = model.predict_proba(df)[0]
        perc = f"{max(neg_perc, pos_perc)*100:.2f}%"

        intro_text = """This is just an expectation from a program and not a certain medical diagnosis. Please check with your doctor. We hope you are always in good health."""

        advice_text = """**WARNING**\n\nStay Calm. Avoid stimulants like caffeine and tobacco. Eat smart and track your blood pressure. Rest and seek medical attention if symptoms worsen."""

        st.write("-" * 50)
        st.write_stream(lambda: (word + " " for word in intro_text.split()))

        if result == 1:
            st.error(f"⚠️ The patient is at risk of stroke! Probability: {perc}")
            st.image("https://media.mehrnews.com/d/2018/11/05/4/2947868.jpg", width=600)
            st.write_stream(lambda: (word + " " for word in advice_text.split()))
        else:
            st.success(f"✅ The patient is not at risk of stroke. Probability: {perc}")
            st.image("https://astrologer.swayamvaralaya.com/wp-content/uploads/2012/08/health1.jpg", width=600)

# ---------------------- Visualizations Tab -------------------------
elif menu == "📊 Visualizations":
    st.sidebar.header("Stroke Risk Visualizations")
    
    @st.cache_resource
    def get_fig_heart_disease(data):
        fig = px.scatter(data, x='heart_disease', y='stroke', trendline="ols",
                         title="Heart Disease vs Stroke Risk")
        return fig

    @st.cache_resource
    def get_fig_BMI(data):
        fig = px.scatter(data, x='bmi', y='stroke', trendline="ols",
                         title="BMI vs Stroke Risk")
        return fig

    @st.cache_resource
    def get_fig_hypertension(data):
        fig = px.scatter(data, x='hypertension', y='stroke', trendline="ols",
                         title="Hypertension vs Stroke Risk")
        return fig

    @st.cache_resource
    def get_fig_glucose(data):
        fig = px.scatter(data, x='avg_glucose_level', y='stroke', trendline="ols",
                         title="Glucose Level vs Stroke Risk")
        return fig

    @st.cache_resource
    def get_fig_age(data):
        fig = plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(data, x='age', hue='stroke', bins=20, alpha=0.4)
        plt.title('Age Histogram')
        plt.subplot(1, 2, 2)
        sns.kdeplot(data=data, x='age', hue='stroke', common_norm=False, alpha=0.2)
        plt.title('Age Density Plot')
        plt.tight_layout()
        return fig

    risk_factor = st.selectbox('Select Risk Factor', ['avg_glucose_level', 'BMI', 'Hypertension', 'Heart Disease', 'Age'])
    if st.button('Show Visualization'):
        if risk_factor == 'Hypertension':
            st.plotly_chart(get_fig_hypertension(data))
        elif risk_factor == 'Heart Disease':
            st.plotly_chart(get_fig_heart_disease(data))
        elif risk_factor == 'BMI':
            st.plotly_chart(get_fig_BMI(data))
        elif risk_factor == 'Age':
            st.pyplot(get_fig_age(data))
        else:
            st.plotly_chart(get_fig_glucose(data))
