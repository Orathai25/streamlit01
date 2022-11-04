import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pickle

html_8="""
<div style="background-color:#EE9513;padding:15px;border-radius:10px 10px 10px 10px;border-style:'double';border-color:black">
<center><h5>การทำนายข้อมูลดอกไม้</h5></center>
</div>
"""

st.markdown(html_8,unsafe_allow_html=True)
st.markdown("")

dt=pd.read_csv('./data/iris.csv')
st.write(dt.head(10))

data1 = dt['sepal.length'].sum()
data2 = dt['sepal.width'].sum()
data3 = dt['petal.length'].sum()
data4 = dt['petal.width'].sum()

dx=[data1, data2, data3, data4]
dx2 = pd.DataFrame(dx, index=['data1','data2','data3','data4'])

st.balloons()

if st.button("แสดงการจิตทัศน์ข้อมูล"):
    st.write(dt.head(20))
    st.bar_chart(dx2)
    st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงข้อมูล") 


html_8="""
<div style="background-color:#EE9513;padding:15px;border-radius:10px 10px 10px 10px;border-style:'double';border-color:white">
<center><h5>การทำนายข้อมูลดอกไม้</h5></center>
</div>
"""

st.markdown(html_8,unsafe_allow_html=True)
st.markdown("")

st_len = st.slider("กรุณาเลือกข้อมูล sepal.length")
sd = st.slider("กรุณาเลือกข้อมูล sepal.width")
pt_len = st.number_input("กรุณาเลือกข้อมูล petal.length")
wd = st.number_input("กรุณาเลือกข้อมูล petal.width")


if  st.button("ทำนายผล"):
    st.markdown("ใส่โมเดล")
    st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงผลการทำนาย")


loaded_model = pickle.load(open('./data/trained_model.sav', 'rb'))
input_data =  (st_len, sd, pt_len, wd)
# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = loaded_model.predict(input_data_reshaped)
st.write(prediction)