import streamlit as st

st.title("Prediction App Layout")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    age = st.number_input("Age", 18, 60)
    salary = st.number_input("Salary", 20000, 200000)

with col2:
    st.subheader("Output")
    if age > 30 and salary > 50000:
        st.success("Prediction: Approved")
    else:
        st.error("Prediction: Not Approved")
