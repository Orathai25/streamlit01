import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pickle

st.image('./image/banner.png')

html_8="""
<div style="background-color:#EE9513;padding:15px;border-radius:10px 10px 10px 10px;border-style:'double';border-color:black">
<center><h5>การทำนายข้อมูลดอกไม้</h5></center>
</div>
"""

st.markdown(html_8,unsafe_allow_html=True)
st.markdown("")

dt=pd.read_csv('./data/train_cus_car.csv')
st.write(dt.head(10))

dt1 = dt['Gender'].sum()
dt2 = dt['Ever_Married'].sum()
dt3 = dt['Age'].sum()
dt4 = dt['Graduated'].sum()
dt5 = dt['Profession'].sum()
dt6 = dt['Work_Experience'].sum()
dt7 = dt['Spending_Score'].sum()
dt8 = dt['Family_Size'].sum()


dx=[dt1, dt2, dt3, dt4, dt5, dt6, dt7, dt8]
dx2 = pd.DataFrame(dx, index=['ddt1, dt2, dt3, dt4, dt5, dt6, dt7, dt8'])


if st.button("แสดงการจิตทัศน์ข้อมูล"):
    st.write(dt.head(20))
    st.bar_chart(dx2)
    st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงข้อมูล") 


html_8="""
<div style="background-color:#EE9513;padding:15px;border-radius:10px 10px 10px 10px;border-style:'double';border-color:white">
<center><h5>การวิเคราะห์ลูกค้าเพื่อแบ่งกลุ่มลูกค้าใหม่</h5></center>
</div>
"""

st.markdown(html_8,unsafe_allow_html=True)
st.markdown("")

gen = st.number_input("กรุณาเลือกข้อมูล เพศ")
mar = st.number_input("กรุณาเลือกข้อมูล สถานะภาพการสมรส")
age = st.number_input("กรุณาเลือกข้อมูล อายุ")
grad = st.number_input("กรุณาเลือกข้อมูล จบการศึกษา")
prof = st.number_input("กรุณาเลือกข้อมูล อาชีพ")
work_e = st.number_input("กรุณาเลือกข้อมูล ประสบการณ์การทำงาน")
spend = st.number_input("กรุณาเลือกข้อมูล อัตราการใช้จ่าย")
fami = st.number_input("กรุณาเลือกข้อมูล สมาชิกในครอบครัว(รวมทั้งตัวลูกค้า)")


if  st.button("ทำนายผล"):
    loaded_model = pickle.load(open('./data/cus_train_model.sav', 'rb'))
    input_data =  (gen, mar, age, grad, prof, work_e, spend, fami)
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    
    st.write(prediction[0])

    st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงผลการทำนาย")


