import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load the pre-trained model
model=tf.keras.models.load_model('mnist_model.h5')

st.title('MNIST Digit Recogniser')
st.write("Upload a 28*28 grayscale image of a digit (0-9)")

uploaded_file=st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image=Image.open(uploaded_file).convert('L') # Convert to grayscale
    invert = st.checkbox("Invert colors (only if digit is white on black)", value=False)

    if invert:
      image = ImageOps.invert(image)

    image=image.resize((28,28)) # Resize to 28x28
    st.image(image,caption='Uploaded Image', use_column_width=True)
    
    image = ImageOps.autocontrast(image)
    
    image_array=np.array(image).astype('float32')/255.0
    image_array=image_array.reshape(1,784) #flatten
    
    prediction=model.predict(image_array)
    prediction_digit= np.argmax(prediction)
    
    st.write("Raw prediction vector:", prediction[0])
    st.write("Predicted index:", np.argmax(prediction))

    
    st.subheader(f'Predicted Digit: {prediction_digit}')
    st.write("Confidence Scores:")
    st.bar_chart(prediction[0])