import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from PIL import ImageChops
from PIL import ImageEnhance
import sqlite3 
import hashlib

# Security
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

# DB Management
conn = sqlite3.connect('data.db')
c = conn.cursor()

# DB  Functions
def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable504(username TEXT,email TEXT,password TEXT)')


def add_userdata(username,email,password):
    c.execute('INSERT INTO userstable504(username,email,password) VALUES (?,?,?)',(username,email,password))
    conn.commit()

def login_user(username,password):
    c.execute('SELECT * FROM userstable504 WHERE username =? AND password = ?',(username,password))
    data = c.fetchall()
    return data

# Load the trained model
model = tf.keras.models.load_model('model_casia_run1.h5')

# Define the labels for prediction
labels = ['Tampered','Authentic']

# Function to preprocess the uploaded image
def preprocess_image(image):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.jpg'
    
    image.save(temp_filename, 'JPEG', quality=90)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    # Resize the image to the required input size
    ela_image = ela_image.resize((128, 128))
    
    # Convert the image to numpy array and normalize the pixel values
    image_array = np.array(ela_image) / 255.0
    
    # Add batch dimension
    preprocessed_image = np.expand_dims(image_array, axis=0)
    
    return preprocessed_image


# Streamlit app
def main():
    st.set_page_config(page_title='Image Forgery Detection', page_icon='logo.png')
    st.title("Image Forgery Detection")
    st.text("Upload an image to detect forgery")

    menu = ["Home","Login","SignUp"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home":
        st.subheader("Home")
    elif choice == "Login":
        st.subheader("Login Section")
        username = st.sidebar.text_input("User Name", key="login-username")
        password = st.sidebar.text_input("Password",type='password',value='', key="login-password")
        if st.sidebar.checkbox("Login"):
            create_usertable()
            hashed_pswd = make_hashes(password)

            result = login_user(username,check_hashes(password,hashed_pswd))
            if result:
                st.success("Logged In as {}".format(username))
                uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
                if uploaded_file is not None:
                    # Display the uploaded image
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
		    
                    # Preprocess the image
                    preprocessed_image = preprocess_image(image)

                    # Make prediction
                    prediction = model.predict(preprocessed_image)
                    prediction_label = labels[np.argmax(prediction)]
                    prediction_confidence = np.max(prediction) * 100

                    # Display the prediction and confidence
                    st.subheader("Prediction:")
                    st.write(prediction_label)
                    st.subheader("Confidence:")
                    st.write(f"{prediction_confidence:.2f}%")
				

    elif choice == "SignUp":
        st.subheader("Create New Account")
        new_user = st.text_input("Username", key="signup-username")
        mailid = st.text_input("Email", key="signup-email")
        new_password = st.text_input("Password",type='password',value='', key="signup-password")

        if st.button("Signup"):
            create_usertable()
            add_userdata(new_user,mailid,make_hashes(new_password))
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")

    

if __name__ == "__main__":
    main()
