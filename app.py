from flask import Flask, render_template, request

from keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)
model = load_model('model_include_disgust_3 CNN layers .h5')

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Resize the image to match the input shape of the model
    image = cv2.resize(image, (48, 48))
    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Normalize the pixel values to be in the range of 0-1
    image = image / 255.0
    # Expand dimensions to match the input shape of the model
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

# Function to perform facial expression recognition
def predict_emotion(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    # Perform prediction using the loaded model
    emotion_probabilities = model.predict(processed_image)[0]
    # Get the index of the emotion with the highest probability
    predicted_emotion_index = np.argmax(emotion_probabilities)
    # List of emotion labels
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    # Get the predicted emotion label
    predicted_emotion = emotion_labels[predicted_emotion_index]
    return predicted_emotion

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file from the request
    uploaded_image = request.files['image']
    # Read the image using OpenCV
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    # Perform facial expression recognition
    predicted_emotion = predict_emotion(image)
    # Render the result template and display the predicted emotion
    return render_template('result.html', emotion=predicted_emotion)

if __name__ == '__main__':
    app.run(debug=True)