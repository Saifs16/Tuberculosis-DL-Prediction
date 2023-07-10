# Importing the necessary libraries
import flask
from flask import render_template
import numpy as np
import io
from PIL import Image

from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras import backend as K

import tensorflow as tf
from tensorflow.keras.models import load_model

# Initialize the Flask app
app = flask.Flask(__name__, template_folder='static')
model = None
response = None

def load_our_model():
    global model
    model = tf.keras.models.load_model('./tb-model.h5')

# Preprocessing the image
def prepare_image(image, target):
    # If the image is not of the RGB format, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize the image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image

# Home Page
@app.route("/")
def index():
    return flask.render_template('index.html')

# Building the predict API endpoint
@app.route("/predict", methods=["POST", "GET"])
def predict():
    # Ensure that the image was loaded properly on our endpoint
    if flask.request.method == "POST" and flask.request.files.get("image"):
        # Read image in PIL format
        image = flask.request.files["image"].read()
        image = Image.open(io.BytesIO(image))
        # Preprocess the image for classification
        image = prepare_image(image, target=(28, 28))
        if image.shape[2] == 1:
            image = np.dstack([image, image, image])
        # Make prediction on the preprocessed image
        image = np.array(image)
        image = image / 255.0

        # Classify the input image and then initialize the list of predictions to return to the client
        prediction = model.predict(np.array(image))

        print("PRED", prediction)
        res = np.argmax(prediction)
        print("RES", res)

        # If res is 0, then the image predicted is Normal else it is Tuberculosis
        # Build the response and return it to the client
        global response
        if res == 0:
            response = {
                'prediction' : {
                    'Normal': 1,
                    'Tuberculosis': 0 
                }
            }
        else:
            response = {
                'prediction' : {
                    'Normal': 0,
                    'Tuberculosis': 1 
                }
            }

        print("response", response)
        return render_template('results.html', response=response)

# If this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":
    print(("* Loading model and starting Flask server..."
        "please wait until server has fully started"))
    load_our_model()
    #app.debug = True
    app.run(host='0.0.0.0')
