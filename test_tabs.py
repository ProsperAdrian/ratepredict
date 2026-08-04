import streamlit as st
import time

st.set_page_config(page_title="Tabs")
t1, t2 = st.tabs(["A", "B"])
with t1:
    st.write("A")
with t2:
    st.write("B")
