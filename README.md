# Image Forgery Detection

This documentation provides an overview and code implementation of a forgery detection model using the ELA (Error Level Analysis) technique and a Convolutional Neural Network (CNN) model. The code is written in Python using the Keras library.

## Table of Contents
- [Introduction](#introduction)
- [Security](#security)
- [Database Management](#database-management)
- [Image Preprocessing](#image-preprocessing)
- [Dataset Preparation](#dataset-preparation)
- [Model Architecture](#model-architecture)
- [Model Training](#model-training)
- [Streamlit App](#streamlit-app)
  - [Home](#home)
  - [Login](#login)
  - [SignUp](#signup)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Note](#note)

## Introduction

Forgery detection is a crucial task in digital forensics that involves distinguishing between authentic and manipulated images.This code is an implementation of an image forgery detection system using Streamlit, TensorFlow, and PIL (Python Imaging Library). The system allows users to upload an image and detect if it has been tampered with or not. It uses in house trained model to make predictions on the uploaded image.The ELA technique calculates the difference in error levels between an original image and a resaved version of the image at a specified compression quality.

## Security

The code includes security measures to protect user passwords. The `make_hashes` function uses the SHA-256 hashing algorithm to generate a hash value from a password string. The `check_hashes` function compares a password with a hashed text to validate the password's correctness.

## Database Management

The code utilizes SQLite to manage a user table for user registration and login. The `create_usertable` function creates a table named `userstable504` if it does not exist. The `add_userdata` function inserts user data (username, email, and hashed password) into the user table. The `login_user` function retrieves user data from the table based on the provided username and password.

## Image Preprocessing

The code defines a `preprocess_image` function to preprocess the uploaded image before making predictions. It uses techniques like Error Level Analysis (ELA) and image enhancement to detect image tampering. The function saves the image as a temporary file, performs ELA to calculate pixel differences, enhances the image brightness, resizes it to the required input size (128x128), converts it to a normalized numpy array, and adds a batch dimension.

The prepare_image function takes an image path and prepares the image for model input. It uses the convert_to_ela_image function to obtain the ELA image, resizes it to a specified image size (e.g., 128x128 pixels), converts it to a NumPy array, and normalizes the pixel values to the range [0, 1]. The prepared image is returned as the output.


## Dataset Preparation

The code prepares a dataset for forgery detection by processing both real and fake images. The real images are obtained from the CASIA dataset, and the fake images are obtained by applying various manipulations to the real images. The `X` list stores the ELA converted images, and the `Y` list stores the corresponding labels (0 for fake, 1 for real). The dataset is split into training and validation sets using the `train_test_split` function from scikit-learn.


## Model Architecture
The forgery detection model is built using a CNN architecture. The `build_model` function defines the sequential model with the following layers:

- Two convolutional layers with 32 filters, kernel size of 5x5, and ReLU activation
- Max pooling layer with a pool size of 2x2
- Dropout layer with a dropout rate of 0.25
- Flatten layer to convert the output to a 1D vector
- Dense layer with 256 units and ReLU activation
- Dropout layer with a dropout rate of 0.5
- Dense layer with 2 units and softmax activation for binary classification (fake or real)


## Model Training

The model is trained using the compiled model with the Adam optimizer and binary cross-entropy loss. The training is performed for a specified number of epochs (e.g., 35) with a batch size of 32. Early stopping is applied to prevent overfitting, using the validation accuracy as the monitored metric. The training history is stored in the `hist` variable. Our trained model has 91% accuracy for train and 94% accuracy for test.


## Streamlit App

The code utilizes Streamlit to create a user interface for the image forgery detection system. It provides a menu with options for Home, Login, and SignUp. The main functionality is in the Login section, where users can enter their credentials and upload an image to detect forgery. The uploaded image is displayed, preprocessed, and fed into the trained model for prediction. The predicted label (Tampered or Authentic) and confidence score are then displayed.

### Home

The Home section provides an overview of the image forgery detection system.

### Login

The Login section allows users to log in with their username and password. If the credentials are correct, they can upload an image for forgery detection. The uploaded image is processed and analyzed, and the prediction results are displayed.

### SignUp

The SignUp section allows new users to create an account by providing a username, email, and password. Upon successful registration, a message is displayed, and users can proceed to the Login menu to log in.

## Usage

To use the image forgery detection system, follow these steps:

1. Make sure you have Python 3.7+ installed.

2. Install the required packages by running the following command: ''' pip install streamlit tensorflow numpy pillow '''

3. Save the code in a file named `image_forgery_detection.py`.

4. Create an SQLite database file named `data.db` in the same directory as the code.

5. Run the code using the following command: '''streamlit run image_forgery_detection.py'''

6. Access the system through your web browser by clicking the provided URL.

7. Select the Login menu and enter your credentials (if already registered) or choose SignUp to create a new account.

8. After logging in, upload an image by clicking the file uploader.

9. The system will preprocess the image, make predictions using a trained model, and display the forgery detection results.

## Dependencies

The code relies on the following libraries and frameworks:

- `streamlit`: Used to create the web-based user interface.
- `matplotlib`: for data visualization
- `keras`: for building and training the CNN model
- `scikit-learn`: for data preprocessing and evaluation metrics
- `tensorflow`: Used to load the trained model and perform predictions.
- `numpy`: Used for array manipulation and calculations.
- `pillow`: Used for image preprocessing and manipulation.
- `sqlite3`: Used for database management and user authentication.
- `hashlib`: Used for password hashing.

Run the following code to install these:
```- pip install streamlit
- pip install matplotlib
- pip install keras
- pip install scikit-learn
- pip install tensorflow
- pip install numpy
- pip install pillow
```

## Note

This code assumes that a trained model named `model_casia_run1.h5` is available in the same directory. Ensure that the model file is present or modify the code to load a different model.Forgery detection using the ELA technique and a CNN model can effectively distinguish between authentic and manipulated images. The trained model can be used to detect forgeries in various applications such as digital forensics and image verification systems.

For any issues or questions, please contact the project contributors mentioned in the code.
