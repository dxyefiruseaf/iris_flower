import streamlit as st
import pickle
import numpy as np

# load model
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Iris Flower Classification")

st.write("Nhap thong so cua hoa Iris")

# input
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

if st.button("Predict"):
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(data)

    species = ["Setosa", "Versicolor", "Virginica"]

    st.success("Loai hoa du doan: " + species[prediction[0]])