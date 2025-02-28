import streamlit as st

x = st.slider("Sélectionner une valeur")
st.write(x, "Son carré est", x * x)
