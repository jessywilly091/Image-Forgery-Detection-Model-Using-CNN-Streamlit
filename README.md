# Image Forgery Detection

## Table of Contents
- [Introduction](#introduction)
- [Security](#security)
- [Database Management](#database-management)
- [Image Preprocessing](#image-preprocessing)
- [Streamlit App](#streamlit-app)
  - [Home](#home)
  - [Login](#login)
  - [SignUp](#signup)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Note](#note)

## Introduction

This code is an implementation of an image forgery detection system using Streamlit, TensorFlow, and PIL (Python Imaging Library). The system allows users to upload an image and detect if it has been tampered with or not. It uses a pre-trained deep learning model to make predictions on the uploaded image.

## Security

The code includes security measures to protect user passwords. The `make_hashes` function uses the SHA-256 hashing algorithm to generate a hash value from a password string. The `check_hashes` function compares a password with a hashed text to validate the password's correctness.

## Database Management

The code utilizes SQLite to manage a user table for user registration and login. The `create_usertable` function creates a table named `userstable504` if it does not exist. The `add_userdata` function inserts user data (username, email, and hashed password) into the user table. The `login_user` function retrieves user data from the table based on the provided username and password.

## Image Preprocessing

The code defines a `preprocess_image` function to preprocess the uploaded image before making predictions. It uses techniques like Error Level Analysis (ELA) and image enhancement to detect image tampering. The function saves the image as a temporary file, performs ELA to calculate pixel differences, enhances the image brightness, resizes it to the required input size (128x128), converts it to a normalized numpy array, and adds a batch dimension.

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
- `tensorflow`: Used to load the trained model and perform predictions.
- `numpy`: Used for array manipulation and calculations.
- `pillow`: Used for image preprocessing and manipulation.
- `sqlite3`: Used for database management and user authentication.
- `hashlib`: Used for password hashing.

## Note

This code assumes that a trained model named `model_casia_run1.h5` is available in the same directory. Ensure that the model file is present or modify the code to load a different model.

For any issues or questions, please contact the project contributors mentioned in the code.
