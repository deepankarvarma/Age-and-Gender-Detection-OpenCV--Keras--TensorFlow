# Age and Gender Detection

This repository contains Python code for an age and gender detection project using the video stream from the camera. The model is trained on the UTKFace dataset, which provides labeled images with age and gender annotations.

## Dataset

The UTKFace dataset used for training and evaluation can be downloaded from the following link: [UTKFace Dataset](https://www.kaggle.com/jangedoo/utkface-new). It consists of facial images of various individuals with age and gender labels.

## Model Architecture

The model architecture used for age and gender detection is based on deep learning techniques. The specific architecture details are mentioned in the code.

## Dependencies

To run the code in this repository, you'll need the following dependencies:

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy

You can install the required packages using `pip`:

```shell
pip install tensorflow keras opencv-python numpy
```

## Usage

1. Clone this repository to your local machine:

```shell
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

2. Download the UTKFace dataset from the provided link and place it in the appropriate directory.

3. Use the provided code to train the age and gender detection model.

4. Run the script to detect age and gender from the video stream:

```shell
python detect_age_gender.py
```

Make sure you have a camera connected to your machine for the live stream.

## Results

The age and gender detection model, trained on the UTKFace dataset, can accurately estimate the age and gender of individuals from the video stream. The code can be further customized or improved to enhance the performance.

## Acknowledgments

- The UTKFace dataset used in this project was sourced from Kaggle: [UTKFace Dataset](https://www.kaggle.com/jangedoo/utkface-new).

## License

This project is licensed under the [MIT License](LICENSE).
