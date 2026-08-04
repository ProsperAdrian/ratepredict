import streamlit as st
import time

with open("production_style.html") as f:
    style = f.read()
    
st.markdown(style, unsafe_allow_html=True)
t1, t2, t3 = st.tabs(["Prediction", "Macro Calendar", "Methodology"])
with t1:
    st.write("A")
with t2:
    st.write("B")
