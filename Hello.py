import streamlit as st
import tensorflow as tf
import numpy as np
from scipy.ndimage.interpolation import zoom
from streamlit_drawable_canvas import st_canvas
from utils import process_image
st.markdown("# Aplicacion de reconocimiento de numeros <3     ¯\_(ツ)_/¯      (☞ﾟヮﾟ)☞")

st.image("https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png?20171214143425")

# Load trained model
model = tf.keras.models.load_model('mi_modelo.h5')

st.write('Dibuja un numero porfis 👉👈:')
# Display canvas for drawing
canvas_result = st_canvas(stroke_width=10, height=28*5, width=28*5)
  
# Process drawn image and make prediction using model
if np.any(canvas_result.image_data):
    # Convert drawn image to grayscale and resize to 28x28
    processed_image = process_image(canvas_result.image_data)
    # Make prediction using model
    prediction = model.predict(processed_image).argmax()
    # Display prediction
    st.header(':red[Resultado:]')
    st.markdown('Creo que este numero es un: \n # :red[' + str(prediction) + ']')
  st.snow()
else:
    # Display message if canvas is empty
    st.header(':red[Resultado:]')
    st.write('Por favor dibuje un numero')
