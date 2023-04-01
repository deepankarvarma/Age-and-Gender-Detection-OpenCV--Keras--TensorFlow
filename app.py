import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# Load the model
model = tf.keras.models.load_model('age_gender_prediction_model.h5')

# Define the image parameters
IMG_WIDTH, IMG_HEIGHT = 200, 200

# Define a function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Define a function to predict the age and gender of an image
def predict_age_gender(image):
    # Preprocess the image
    image = preprocess_image(image)
    # Predict the gender (0 = male, 1 = female)
    gender_pred = 'Female' if model.predict(image)[0] > 0.5 else 'Male'
    # Predict the age
    age_pred = int(model.predict(image) * 100)
    return gender_pred, age_pred


# Define the Streamlit app
def app():
    # Set the app title
    st.title('Age and Gender Prediction App')

    # Upload an image
    image_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
    if image_file is not None:
        # Read the image file and convert it to an array
        image = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # Display the image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        # Predict the age and gender
        gender_pred, age_pred = predict_age_gender(image)
        # Display the results
        st.write('Gender Prediction: ', gender_pred)
        st.write('Age Prediction: ', age_pred)

if __name__ == '__main__':
    app()