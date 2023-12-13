# Facial Expression Recognition App

This is a simple web application built with Flask for performing facial expression recognition using a pre-trained convolutional neural network (CNN) model.

## Prerequisites

Make sure you have the following dependencies installed:

- Python (3.6 or higher)
- Flask
- Keras
- OpenCV
- Numpy

You can install these dependencies using the following command:

```bash
pip install flask keras opencv-python numpy


Usage

1. Clone the repository
2. Run: cd facial-expression-recognition-app to enter the project folder
3. Place your trained Keras model file (model_include_disgust_3 CNN layers .h5) in the project directory.
4. Run the Flask application: python app.py
5. Open your web browser and go to http://127.0.0.1:5000/. You should see the home page of the facial expression recognition app.
6. Upload an image containing a face, and the app will predict the facial expression.
