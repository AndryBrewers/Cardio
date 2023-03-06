import streamlit as st
from pickle import dump, load
import pickle
import pandas as pd
from pandas import DataFrame


st.header('Предсказание сердечных заболеваний')
def load():
    with open('C:/Users/User/Desktop/Kaggle/rscv_gbr.pcl', 'rb') as fid:
        return pickle.load(fid)




age = st.slider('Возраст, дней', 0, 110, key='age') 
gender = st.radio('Выберите пол', options=('M','Ж'), key='gender') 
height = st.slider('Рост, см', 0, 200, key='height')  
weight = st.slider('Вес, кг', 0, 200, key='weight')
ap_hi = st.slider('Вернее давление', 60, 240, key='ap_hi')
ap_lo = st.slider('Нижнее давление', 40, 110, key='ap_lo')
cholesterol = st.radio('Уровень хлолестирина', options=(1, 2, 3), key='cholesterol')
gluc = st.radio('Уровень глюкозы', options=(1, 2, 3), key='gluc')
smoke = st.radio('Вы курите?', options=('Нет', 'Да'), key='smoke')
alco = st.radio('Вы употребляете алкоголь?', options=('Нет', 'Да'), key='alco')
active = st.radio('Вы ведете активный образ жизни?', options=('Нет', 'Да'), key='active')

age=age*365
if gender == 'M':
    gender=1
else:
    gender=2

if smoke == 'Нет':
    smoke=0
else:
    smoke=1

if alco == 'Нет':
    alco=0
else:
    alco=1

if active == 'Да':
    active=1
else:
    active=0

data = {'age': [age], 'gender': [gender], 'height': [height], 'weight':[weight], 'ap_hi':[ap_hi], 'ap_lo':[ap_lo], 'cholesterol':[cholesterol], 'gluc':[gluc], 'smoke':[smoke], 'alco':[alco], 'active':[active]}
df = pd.DataFrame(data)

#df = pd.DataFrame(age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, columns=['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'])

rscv_gbr = load()


gbr_probs = rscv_gbr.predict_proba(df)[:, 1]
st.write(gbr_probs)