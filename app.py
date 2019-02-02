import os

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf

#from data.dogid.mypup import *
from data.digits.predict import *
import re
import base64
import cv2
import io

UPLOAD_FOLDER = 'img/uploads/mdb'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


app = Flask(__name__)

global graph
graph = tf.get_default_graph()

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# homepage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/digits')
def api_root():
    return render_template('mydigits.html')

@app.route('/predict', methods = ['POST'])
def api_predict():
    # Get the uploaded image
    img = request.files['file']
    
    # Store in memory
    in_memory_file = io.BytesIO()
    img.save(in_memory_file)
    
    # Convert to array representing black-and-white image
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, 0)/255
    
    # Reshape for the model
    data = img.reshape(-1, 28,28,1)
    
    # The model is trained on white-on-black images. If a white-on-black image
    # is used, it will likely give incorrect results. Here is a basic fix to
    # try to detect black-on-white images and convert them. Probably not fool-proof.
    if ( data.mean() > 0.5 ):
        data = ValueInvert(data)
        
    # Generate prediction
    pred = mnist_model.predict(data)
    
    # Provide response
    return "The image you uploaded shows a " + str(pred) + ".\n"

@app.route('/post-data-url', methods = ['POST'])
def api_predict_from_dataurl():
    # In this case, we read the image data from a base64 data URL
    imgstring = request.form.get('data')
    
    # Stripping the metadata
    imgstring = imgstring.replace("data:image/png;base64,", "")
    
    # Decoding
    imgdata = base64.b64decode(imgstring)
    
    # Unfortunately I have not yet figured out how to get this data into
    # a usable format directly, without first saving to file
    filename = 'canvas_image.png'
    with open(filename, 'wb') as f:
        f.write(imgdata)
    
    # Reading the image into OpenCV
    img = cv2.imread('canvas_image.png', 0)
    
    # Resize to fit the neural network
    imgsmall = cv2.resize(img, (28,28))
    
    # Save the resized image for manual checking
    cv2.imwrite("resized.png", imgsmall)
    
    # Reshape for the model
    data = imgsmall.reshape(-1, 28,28,1)/255
    
    # Convert to white-on-black
    data = ValueInvert(data)
    
    # Generate prediction
    pred = mnist_model.predict(data)
    
    # Return the prediction
    return str(np.argmax(pred[0]))

# Takes an array, assumed to contain values between 0 and 1, and inverts
# those values with the transformation x -> 1 - x.
def ValueInvert(array):
    # Flatten the array for looping
    flatarray = array.flatten()
    
    # Apply transformation to flattened array
    for i in range(flatarray.size):
        flatarray[i] = 1 - flatarray[i]
        
    # Return the transformed array, with the original shape
    return flatarray.reshape(array.shape)


if __name__ == '__main__':
#    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'

    app.run(debug=False, host='0.0.0.0',port="50001")