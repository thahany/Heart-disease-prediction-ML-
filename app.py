import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

df=pd.read_csv('heart.csv')

standardScaler = StandardScaler()
X = df.drop(['target'],axis=1)
y = df['target'].values
X_trans = standardScaler.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test,y_train, y_test = train_test_split(X_trans,y, test_size = 0.3, random_state = 2021)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=10)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
from PIL import Image
#pickle_in = open('covid19R_model.pkl', 'rb')
#classifier = pickle.load(pickle_in)

def welcome():
    return 'Stay_Home Stay_Safe'

def predict_class(df):
    pred = model.predict(df)
    return pred

def main():
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Heart Disease Prediction</h2>
    </div>
    """
  
    st.markdown(html_temp , unsafe_allow_html= True)
    image = Image.open('hdp.png')
    # st.image(image, use_column_width=True,format='png')
    st.image(image, use_column_width=True)
    
    
    
    age=st.text_input('Age')
    # age=st.slider('Age', 0,100)
    sex=st.text_input('Sex 1(Male)-0(Female)',' ')
    cp=st.text_input('Chest pain (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 4 = asymptomatic)',' ')
    trestbps=st.text_input('Resting blood pressure in mm Hg, Range;90-200')
    # trestbps=st.slider('Resting blood pressure in mm Hg ', 90,200)
    chol=st.text_input('Serum cholesterol in mg/dl, Range;126-564')
    # chol=st.slider('Serum cholestoral in mg/dl', 126,564)
    fbs=st.text_input('Fasting blood sugar > 120 mg/dl, (1 = true; 0 = false)',' ')
    restecg=st.text_input('Resting electrocardiographic results (0 = normal; 1 = having ST-T; 2 = hypertrophy)','')
    # restecg=st.slider('Resting electrocardiographic results (0 = normal; 1 = having ST-T; 2 = hypertrophy)','')
    thalach=st.text_input('Maximum heart rate achieved, Range;65-210')
    # thalach=st.slider('Maximum heart rate achieved', 65,210)
    exang=st.text_input('Exercise induced angina 1 = yes; 0 = No',' ')
    # oldpeak=st.text_input('ST depression induced by exercise relative to rest (0-7)',' ')
    slope=st.text_input('The slope of the peak exercise ST segment (1 = upsloping; 2 = flat; 3 = downsloping)', ' ')
    ca=st.text_input('Number of major vessels (0-3) colored by flouroscopy', ' ')
    thal=st.text_input('Hemoglobin level (1 = normal, 2 = fixed defect, 3 = defect)', ' ')
              
    result =""
    if st.button('Predict'):
        # dataset = [int(age),int(sex),int(cp),int(trestbps),int(chol),int(fbs),int(restecg),
        #                       int(thalach),int(exang),float(oldpeak),int(ca),int(thal)]
        dataset = [int(age),int(sex),int(cp),int(trestbps),int(chol),int(fbs),int(restecg),
                              int(thalach),int(exang),int(slope),int(ca),int(thal)]
        df = pd.DataFrame(dataset)
        df_trans = standardScaler.fit_transform(df)
        df_trans = df_trans.T
        result=model.predict(df_trans)
        if result==0:
             result="Your heart health is good, Enjoy"
        else:
             result="Sorry you Have symptoms of Heart Disease"
            
  
    st.success('{}'.format(result))
    if st.button("About"):
        st.text("BMG 5111F_EHR Project_Heart Disease Prediction.")
        # st.text("A controlled carbohydrate lifestyle really prevents risk factors for heart disease.")
        # st.text("Github link: https://github.com/ayushkesh/Heart-disease-prediction-ML- ")   
        
    html_temp1 = """
    <div style="background-color:blue;"> 
    <p style="color:white;text-align:center;" >By: <b>Raouf & Thahany</b> </p>
    </div>
    """
    st.markdown(html_temp1,unsafe_allow_html=True)
if __name__ == '__main__':
    
    main()
        
     