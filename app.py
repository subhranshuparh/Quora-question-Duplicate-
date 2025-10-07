import streamlit as st
import helper
from joblib import load

# Load model safely with memory mapping
model = load('model.joblib', mmap_mode='r')

st.title("🔍 Duplicate Question Detector")
st.write("This app checks whether two questions have the same meaning or not.")

q1 = st.text_input("Enter Question 1:")
q2 = st.text_input("Enter Question 2:")

if st.button("Find"):
    if not q1.strip() or not q2.strip():
        st.warning("⚠️ Please enter both questions.")
    else:
        try:
            query = helper.query_point_creator(q1, q2)
            result = model.predict(query)[0]

            if result:
                st.success("✅ The questions are **Duplicate**")
            else:
                st.error("❌ The questions are **Not Duplicate**")
        except Exception as e:
            st.error(f"An error occurred: {e}")
