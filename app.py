import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
# Load the trained model
model = tf.keras.models.load_model('age_gender_prediction_model.h5')

# Define the labels for gender prediction
gender_labels = ['Male', 'Female']

# Define the function to make predictions
def predict_age_and_gender(image):
    # Resize the image
    resized_image = cv2.resize(image, (200, 200))
    # Convert to float32
    image = np.float32(resized_image)
    # Normalize the image
    image /= 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    # Make the prediction
    prediction = model.predict(image)
    # Get the predicted age
    age = int(prediction[0])
    # Get the predicted gender
    gender_prediction = model.predict_classes(image)
    gender = gender_labels[gender_prediction[0][0]]
    # Return the predicted age and gender
    return age, gender

# Define the Streamlit app
def app():
    # Set the app title
    st.set_page_config(page_title='Age and Gender Prediction', page_icon='👴👩')
    st.title('Age and Gender Prediction')
    
    # Allow the user to upload an image
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        # Load the image
        image = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # Make the prediction
        age, gender = predict_age_and_gender(image)
        # Display the image and prediction
        st.image(image, caption=f'Predicted Age: {age}, Predicted Gender: {gender}', use_column_width=True)

# Run the Streamlit app
if __name__ == '__main__':
    app()