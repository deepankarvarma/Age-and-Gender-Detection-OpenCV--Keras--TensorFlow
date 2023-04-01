import tensorflow as tf
import cv2
import numpy as np

# Load the saved model
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
    age_pred = np.argmax(model.predict(image))
    return gender_pred, age_pred

# Load an image and predict the age and gender
image_path = 'test1.jpg'  # replace with your own image path
image = cv2.imread(image_path)
gender_pred, age_pred = predict_age_gender(image)
print('Gender Prediction:', gender_pred)
print('Age Prediction:', age_pred)