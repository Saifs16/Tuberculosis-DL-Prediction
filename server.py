# Importing the necessary libraries
import flask
from flask import render_template
import numpy as np
import io
from PIL import Image
import cv2

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import tensorflow as tf

# Initialize the Flask app
app = flask.Flask(__name__, template_folder='static')
model = None
response = None

def load_our_model():
    global model
    model = tf.keras.models.load_model('tb-model.h5')

# Preprocessing the image
def prepare_image(image, target_size):
    # If the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize the image to the target size
    image = image.resize(target_size)
    image = np.array(image)
    
    # Ensure all images have 3 channels
    if len(image.shape) == 2:
        image = np.stack([image]*3, axis=-1)
        
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image/255.
    
    # Return the processed image
    return image

@app.route("/")
def index():
    return flask.render_template('index.html')

@app.route("/predict", methods=["POST", "GET"])
def predict():
    data = {}  # dictionary to store results 
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            image = prepare_image(image, target_size=(28, 28))

            # Predict the disease using the preprocessed image
            predictions = model.predict(np.array([image]))

            results = np.argmax(predictions, axis=1)

            global response
            
            if results[0] == 0:
                print('Predicted to be Normal')
                response = {
                    'prediction' : {
                        'Normal': 1,
                        'Tuberculosis': 0 
                    }
                }
            elif results[0] == 1:
                print('Predicted to be Tuberculosis')
                response = {
                    'prediction' : {
                        'Normal': 0,
                        'Tuberculosis': 1 
                    }
                }

        return render_template('results.html', response=response)

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_our_model()
    app.run(debug=True)