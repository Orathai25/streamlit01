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
<center><h5>การวิเคราะห์ลูกค้าเพื่อแบ่งกลุ่มลูกค้าใหม่</h5></center>
</div>
"""

st.markdown(html_8,unsafe_allow_html=True)
st.markdown("")

dt=pd.read_csv('./data/train_cus_car.csv')
st.write(dt.head(10))

data1 = dt['Gender'].sum()
data2 = dt['Ever_Married'].sum()
data3 = dt['Age'].sum()
data4 = dt['Graduated'].sum()
data5 = dt['Profession'].sum()
data6 = dt['Work_Experience'].sum()
data7 = dt['Spending_Score'].sum()
data8 = dt['Family_Size'].sum()


dx=[data1, data2, data3, data4, data5, data6, data7, data8]
dx2 = pd.DataFrame(dx, index=['data1','data2','data3','data4','data5','data6','data7','data8'])


if st.button("แสดงการจิตทัศน์ข้อมูล"):
    st.write(dt.head(20))
    st.plotly_chart(dx2)
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

gen = st.radio("กรุณาเลือกข้อมูล เพศ",('0','1'))
if gen == '0':
    st.write('เพศชาย.')
elif gen == '1':
    st.write('เพศหญิง.')
else:
    st.write("คุณยังไม่ได้เลือกเพศ.")

mar = st.radio("กรุณาเลือกข้อมูล สถานะภาพ",('0','1'))
if mar == '0':
    st.write('ยังไม่แต่งงาน.')
elif mar == '1':
    st.write('แต่งง่นแล้ว.')
else:
    st.write("คุณยังไม่ได้กรอกสถานะ.")

age = st.number_input("กรุณาเลือกข้อมูล อายุ")

grad = st.radio("กรุณาเลือกข้อมูล การศึกษา",('0','1'))
if grad == '0':
    st.write('ยังไม่สำเร็จการศึกษา.')
elif grad == '1':
    st.write('สำเร็จการศึกษาแล้ว.')
else:
    st.write("คุณยังไม่ได้ระบุสถานะการศึกษา.")

st.write('อาชีพ: 0 Healthcare, 1 Engineer, 2 Lawyer, 3 Entertainment, 4 Artist, 5 Executive, 6 Doctor, 7 Homemaker, 8 Marketing')
prof = st.selectbox('อาชีพของคุณ',('0', '1', '2','3', '4', '5','6', '7', '8'))
st.write('อาชีพ:', prof)

work_e = st.number_input("กรุณาเลือกข้อมูล ประสบการณ์การทำงาน")

st.write('อัตราการใช้จ่าย: 0 ต่ำ, 1 กลาง, 2 สูง')
spend = st.selectbox('อัตราการใช้จ่ายของคุณ',('0', '1', '2'))
st.write('อัตราการใช้จ่าย:', prof)

fami = st.number_input("กรุณาเลือกข้อมูล สมาชิกในครอบครัว(รวมทั้งตัวลูกค้า)")


if  st.button("ทำนายผล"):
    loaded_model = pickle.load(open('./data/cus_seg_model.sav', 'rb'))
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


