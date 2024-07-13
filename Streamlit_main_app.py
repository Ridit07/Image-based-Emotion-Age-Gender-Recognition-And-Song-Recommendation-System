

import streamlit as st
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing import image
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Load emotion detection model
emotion_model = load_model("ferNet.h5")

# Load age and gender detection model
age_gender_model = load_model("agegender.h5")

# Load face cascade classifier
face_haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load music dataset
mood_music = pd.read_csv(
    "data_moods.csv")
mood_music = mood_music[['name', 'artist', 'mood']]

# Function to detect emotion, age, and gender
def detect_emotion(test_img):
    # Convert image to grayscale
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        # Extract face region
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        # Predict emotion
        emotion_predictions = emotion_model.predict(img_pixels)
        max_index = np.argmax(emotion_predictions[0])
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        detected_emotion = emotions[max_index]

        # Extract face for age and gender prediction
        face_img = gray_img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (128, 128))
        face_img = face_img.reshape(1, 128, 128, 1) / 255.0

        # Predict age and gender
        age, gender = age_gender_model.predict(face_img)
        age = age *100
        print(age)
        print(gender)
        predicted_age = int(age[0][0])
        
        predicted_gender = "Male" if gender[0][0] > 0.5 else "Female"
    
        # Display the detected features on the Streamlit UI
        st.subheader(f"Detected Emotion: {detected_emotion}")
        st.subheader(f"Predicted Age: {predicted_age}")
        st.subheader(f"Predicted Gender: {predicted_gender}")

        # Suggest music based on detected emotion
        suggest_music(detected_emotion)

    return test_img


def suggest_music(emotion):
    if emotion in ['angry', 'disgust', 'fear']:
        filter_condition = mood_music['mood'] == 'Calm'
    elif emotion in ['happy', 'neutral']:
        filter_condition = mood_music['mood'] == 'Happy'
    elif emotion == 'sad':
        filter_condition = mood_music['mood'] == 'Sad'
    elif emotion == 'surprise':
        filter_condition = mood_music['mood'] == 'Energetic'
    else:
        # Handle other cases as needed
        return

    filtered_music = mood_music[filter_condition]
    if not filtered_music.empty:
        suggested_music = filtered_music.sample(n=5)
        suggested_music.reset_index(inplace=True)
        st.subheader("Suggested Music:")
        st.table(suggested_music)
    else:
        st.warning(f"No music found for emotion: {emotion}")


def main():
    st.title("Facial Emotion Analysis and Music Suggestion")

    # Create buttons for live feed and image upload
    if st.button("Process Video Feed"):
        process_video_feed()
    elif st.button("Upload Image"):
        process_uploaded_image()

def process_video_feed():
    cap = cv2.VideoCapture(0)
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error capturing video feed.")
            break

        # Display the video feed with detected emotion
        st.image(detect_emotion(frame), channels="BGR", use_column_width=True)

        frame_counter += 1
        if frame_counter >= 5:
            break

    # Release the video capture
    cap.release()

def process_uploaded_image():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Convert to grayscale
        gray_img = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

        # Display the uploaded image
        st.image(opencv_image, channels="BGR", use_column_width=True)

        # Process the image for emotion, age, and gender detection
        detect_emotion(opencv_image)


if __name__ == "__main__":
    main()
