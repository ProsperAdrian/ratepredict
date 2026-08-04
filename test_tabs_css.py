import streamlit as st
import time

st.markdown("""
<style>
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
    background: red !important;
    visibility: hidden !important;
    opacity: 0 !important;
}
</style>
""", unsafe_allow_html=True)
t1, t2 = st.tabs(["A", "B"])
with t1:
    st.write("A")
with t2:
    st.write("B")
