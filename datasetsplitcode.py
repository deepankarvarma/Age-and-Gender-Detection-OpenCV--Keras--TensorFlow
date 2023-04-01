import os
import shutil

# Define the age ranges or bins
age_ranges = [(0, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 79), (80, 89), (90, 99), (100, 116)]

# Define the gender categories
genders = ['male', 'female']

# Create the age and gender directories
for gender in genders:
    for age_range in age_ranges:
        folder_name = f"age_{age_range[0]}-{age_range[1]}"
        os.makedirs(os.path.join('./data/UTKFace', gender, folder_name), exist_ok=True)

# Loop through each file in the dataset
for file_name in os.listdir('./data/UTKFace'):

    try:
        # Extract the age and gender values from the file name
        age, gender = file_name.split('_')[:2]
        age = int(age)

        # Determine which age range or bin the image belongs to
        for i, age_range in enumerate(age_ranges):
            if age >= age_range[0] and age <= age_range[1]:
                age_folder = f"age_{age_range[0]}-{age_range[1]}"
                break

        # Determine which gender category the image belongs to
        if gender == '0':
            gender_folder = genders[0]  # male
        else:
            gender_folder = genders[1]  # female

        # Move the image to the corresponding folder or class
        source_path = os.path.join('./data/UTKFace', file_name)
        destination_path = os.path.join('./data/UTKFace', gender_folder, age_folder, file_name)
        shutil.move(source_path, destination_path)
    except:
        # Skip any files that don't follow the naming convention
        pass