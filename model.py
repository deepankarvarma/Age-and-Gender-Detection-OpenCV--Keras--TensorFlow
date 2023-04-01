import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the input size for the model
input_shape = (224, 224, 3)

# Define the number of age classes
num_age_classes = 11

# Define the number of gender classes
num_gender_classes = 2

# Define the input layer
input_layer = Input(shape=input_shape)

# First convolutional block
x = Conv2D(32, (3,3), activation='relu', padding='same')(input_layer)
x = MaxPooling2D((2,2))(x)
x = BatchNormalization()(x)

# Second convolutional block
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)
x = BatchNormalization()(x)

# Third convolutional block
x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)
x = BatchNormalization()(x)

# Flatten the output
x = Flatten()(x)

# Gender classification branch
gender_branch = Dense(64, activation='relu')(x)
gender_branch = Dropout(0.5)(gender_branch)
gender_branch = Dense(num_gender_classes, activation='softmax', name='gender_output')(gender_branch)

# Age regression branch
age_branch = Dense(64, activation='relu')(x)
age_branch = Dropout(0.5)(age_branch)
age_branch = Dense(32, activation='relu')(age_branch)
age_branch = Dropout(0.5)(age_branch)
age_branch = Dense(num_age_classes, activation='softmax', name='age_output')(age_branch)

# Define the multi-output model
model = Model(inputs=input_layer, outputs=[gender_branch, age_branch])

# Compile the model
model.compile(optimizer='adam',
              loss={'gender_output': 'categorical_crossentropy', 'age_output': 'categorical_crossentropy'},
              metrics={'gender_output': 'accuracy', 'age_output': 'mae'})

# Create the ImageDataGenerator for training data
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    './data/UTKFace',
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='categorical',
    subset='training')

# Create the ImageDataGenerator for validation data
val_generator = train_datagen.flow_from_directory(
    './data/UTKFace',
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='categorical',
    subset='validation')

# Train the model
history = model.fit(train_generator, epochs=2, validation_data=val_generator)

# Save the model
model.save('age_gender_classification_model.h5')
