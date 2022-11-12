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





