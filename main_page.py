import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time as time

st.title('Stroke Prediction System 🧠')
image_url = "https://my.clevelandclinic.org/-/scassets/images/org/patient-experience/patient-stories/173-advanced-stroke-procedure-saves-patient-after-deep-brain-bleed/deep-brain-bleeds-new-2.gif"
st.image(image_url, caption="Real-time visualization of a deep brain stroke — emphasizing the urgency of early detection and prevention.", use_container_width=True)


data_location = "df_cleaned.csv"

@st.cache_data
def load_data(data_location):
    data = pd.read_csv(data_location)
    return data

data = load_data(data_location)
columns_to_drop = ['work_type', 'Residence_type']
data.drop(columns=columns_to_drop, axis=1, inplace=True)
X = data.drop('stroke', axis=1)  # split features
y = data['stroke']  # split target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train) 
y_pred = model.predict(X_test) 
accuracy = accuracy_score(y_pred, y_test)


with st.form(key='prediction_form'):
    st.subheader('Select your features')

    col1, col2 = st.columns(2)
    with col1:
        sex = st.selectbox('Select Gender', ['male', 'female'])
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        ever_married = st.selectbox('Select Ever Married', ['Yes', 'No'])
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        smoking_status = st.selectbox('Select Smoking Status', ['Yes', 'Never smoke'])
        st.write("")
        st.write("")
        st.write("")
        st.write("")
    with col2:
        hypertension = st.selectbox('Select Hypertension', ['Ever had hypertension', 'Not Hade hypertension'])
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        heart_disease = st.selectbox('Select Heart Disease', ['Ever had heart_disease', 'Not Hade heart_disease'])

    col6, col7, col8 = st.columns(3)
    with col6:
        age = st.slider("Select Age", min_value=1, max_value=100, step=1, value=50)
    with col7:
        bmi = st.slider("Select BMI", min_value=10, max_value=50, step=1, value=30)
    with col8:
        avg_glucose_level = st.slider("Select Avg Glucose Level", min_value=50, max_value=270, step=1, value=150)

    submit_button = st.form_submit_button(label='Confirm')


df = pd.DataFrame({
    'sex': [sex],
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'ever_married': [ever_married],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'smoking_status': [smoking_status]
}, index=[0])

def transform_sex(df):
    return 1 if df['sex'] == 'male' else 0

def transform_ever_married(df):
    return 1 if df['ever_married'] == 'Yes' else 0    

def transform_smoking_status(df):
    return 0 if df['smoking_status'] == 'Never smoke' else 1

def transform_hypertension(df):
    return 1 if df['hypertension'] == 'Ever had hypertension' else 0  

def transform_heart_disease(df):
    return 1 if df['heart_disease'] == 'Ever had heart_disease' else 0     


df['sex'] = df.apply(transform_sex, axis=1)
df['ever_married'] = df.apply(transform_ever_married, axis=1)
df['smoking_status'] = df.apply(transform_smoking_status, axis=1)
df['hypertension'] = df.apply(transform_hypertension, axis=1)
df['heart_disease'] = df.apply(transform_heart_disease, axis=1)

result=model.predict(df)

neg_perc=model.predict_proba(df)[0][0]
pos_perc=model.predict_proba(df)[0][1]

perc=''
if neg_perc>pos_perc:
    perc= str(neg_perc*100)+'%'
else:
    perc= str(pos_perc*100)+'%'


text_result=''
if result ==1:
   text_result="⚠️ The patient is at risk of stroke!"
else:
    text_result="✅ The patient is not at risk of stroke."
start_text = f"""
This is just an expectation
from a program and not sure 
 please check with your doctor
but this program expects that
We hope that you are always
in good health \n\n

"""
st.write("-"*50)
end_text="""**WARNING**\n


It’s essential to stay calm and take immediate steps to manage your condition until you can see a doctor. Here are some things you can do:\n

Relax: Try to stay calm and take deep breaths. Stress can increase blood pressure.\n


Avoid Stimulants: Stay away from caffeine, nicotine, and alcohol, as these can raise your blood pressure.\n

Stay Hydrated: Drink plenty of water to help maintain healthy blood pressure levels.\n

Monitor Your Blood Pressure: If you have a blood pressure monitor, keep track of your readings.\n

Eat Healthily: Focus on fruits, vegetables, and low-sodium foods. Avoid processed foods and excess salt.\n

Rest: Take breaks and avoid heavy physical activity.\n

Seek Medical Attention if Necessary: If you experience symptoms like chest pain, severe headache, dizziness, or shortness of breath, seek immediate medical help.

"""   
def stream_data():
    for word in start_text.split(" "):
        yield word + " "
        time.sleep(0.02)

    if result == 1:
        st.error(f"⚠️ The patient is at risk of stroke! with {perc}")
        
       
        st.image("https://media.mehrnews.com/d/2018/11/05/4/2947868.jpg", width=600)

        
        for word in end_text.split(" "):
            yield word + " "
            time.sleep(0.8)

    else:
        st.success(f"✅ The patient is not at risk of stroke. with {perc}")
        st.image("https://astrologer.swayamvaralaya.com/wp-content/uploads/2012/08/health1.jpg", width=600)








if submit_button:
    st.write_stream(stream_data)
    
# if submit_button:
#     result = model.predict(df)
#     st.write(df.head())
#     if result == 0:
#         st.success("✅ The patient is not at risk of stroke.")
#         st.image("https://astrologer.swayamvaralaya.com/wp-content/uploads/2012/08/health1.jpg", width=600)
#     else:
#         st.error("⚠️ The patient is at risk of stroke!")
#         st.image("https://media.mehrnews.com/d/2018/11/05/4/2947868.jpg", width=600)
#     st.write("")


st.sidebar.header(" stroke visualizations")

# ✅ تخزين الرسومات باستخدام caching لتسريع الأداء

@st.cache_resource
def get_fig_heart_disease(data):
    fig = px.scatter(data, x='heart_disease', y='stroke', trendline="ols",
                     title="Relationship between Heart Disease and Stroke Risk",
                     color_discrete_sequence=["#ff7f0e"])
    fig.update_layout(title=dict(font=dict(size=20, color='blue'), x=0.5, xanchor='center', pad=dict(t=20)),
                      xaxis_title="Heart Disease (0 = No, 1 = Yes)", yaxis_title="Stroke Risk Probability")
    return fig

@st.cache_resource
def get_fig_hypertension(data):
    fig = px.scatter(data, x='hypertension', y='stroke', trendline="ols",
                     title="Relationship between Hypertension and Stroke Risk",
                     color_discrete_sequence=["#ff7f0e"])
    fig.update_layout(title=dict(font=dict(size=20, color='blue'), x=0.5, xanchor='center', pad=dict(t=20)),
                      xaxis_title="Hypertension (0 = No, 1 = Yes)", yaxis_title="Stroke Risk Probability")
    return fig

@st.cache_resource
def get_fig_glucose(data):
    fig = px.scatter(data, x='avg_glucose_level', y='stroke', trendline="ols",
                     title="Relationship between avg_glucose_level and Stroke Risk",
                     color_discrete_sequence=["#ff7f0e"])
    fig.update_layout(title=dict(font=dict(size=20, color='blue'), x=0.5, xanchor='center', pad=dict(t=20)),
                      xaxis_title="avg_glucose_level", yaxis_title="Stroke Risk Probability")
    return fig

@st.cache_resource
def get_fig_age(data):
    fig = plt.figure(figsize=(8, 5))
    plt.subplot(1, 2,1)
    sns.histplot(data=data, x='age', hue='stroke', element='poly',bins=20, alpha=0.2)
    plt.title('Histogram of bmi',color="#D71313", fontsize = 13, pad=10)
    plt.subplot(1, 2,2)
    sns.kdeplot(data=data, x='age', hue='stroke', common_norm=False,alpha=0.2)
    plt.title('Density Distribution of age',color="#D71313", fontsize = 13, pad=10)
    fig.suptitle('age: Histogram and Density Distribution',fontsize=18)
    plt.subplots_adjust(left=0.2,right=0.3)
    plt.tight_layout()
    return fig

risk_factor = st.sidebar.selectbox('Select Risk Factor', ['Hypertension', 'Heart Disease','avg_glucose_level','age'])
con = st.sidebar.button('Confirm')

if con:
    if risk_factor == 'Hypertension':
        st.plotly_chart(get_fig_hypertension(data), use_container_width=True)
        st.write("The line show a positive correlation. As hypertension increases, the probability of having a stroke also increases")
    elif risk_factor == 'Heart Disease':
        st.plotly_chart(get_fig_heart_disease(data), use_container_width=True)
        st.write("The line show a positive correlation. As heart_disease increases, the probability of having a stroke also increases")
    elif risk_factor =='age':
        st.pyplot(get_fig_age(data))
        st.write("Stroke cases (stroke=1, orange) are more prevalent in middle-aged and older individuals (40+ years), aligning with known medical research that stroke risks increase with age. The density plot further confirms that stroke cases (stroke=1) peak around 50 to 70 years, This suggests that stroke is more common in older populations compared to younger individuals.")
    else:
        st.plotly_chart(get_fig_glucose(data), use_container_width=True)
        st.write("The line show a positive correlation. As avg_glucose_level increases, the probability of having a stroke also increases")
