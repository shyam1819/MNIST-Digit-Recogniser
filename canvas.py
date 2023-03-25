import pandas as pd
import numpy as np
#from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
#import pickle
import joblib
import cv2

stroke_width=13
stroke_color="#FFFFFF"
bg_color="#000000"

canvas_result = st_canvas(
    #fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    #background_image=Image.open(bg_image) if bg_image else None,
    #update_streamlit=realtime_update,
    height=280,
    width=280,
    drawing_mode="freedraw",
    #point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas"
)


img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
#rescaled = cv2.resize(img, (280, 280), interpolation=cv2.INTER_NEAREST)

model=tf.keras.models.load_model("MNIST_NN.model")


if st.button("Predict"):
    #rescaled=cv2.resize(rescaled,(28,28))
    #im= cv2.cvtColor(rescaled, cv2.COLOR_BGR2GRAY)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    prediction_knn=knn.predict(img.reshape(1,784))[0]
    img=tf.keras.utils.normalize([img],axis=1)
    prediction_NN=model.predict(img)[0]
    #st.write(f'Prediction by KNN model: {prediction_knn}')
    st.write(f"Prediction by a Neural Network: {np.argmax(prediction_NN)}")
    


