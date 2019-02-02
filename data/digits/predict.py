from keras.preprocessing import image
from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

mnist_model = load_model('data/digits/models/mnist_cnn_model.h5')
mnist_model._make_predict_function()

def predict_user_img(img_path, model):
    
    # takes user image as input, preprocesses it,
    # and returns the prediction
    # example use: predict_user_img('digit3.png', 'mnist_model')
    
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (28, 28))
    img = cv2.bitwise_not(img)
    
    if(model == 'model_1'):
        x = img.reshape(1,784).astype('float32')
        probs = model_1.predict(x)
        prediction = np.argmax(probs)
        return prediction, plt.imshow(img)
    
    elif(model == 'mnist_model'):
        x = img.reshape(1, 28, 28,1).astype('float32')
        probs = mnist_model.predict(x)
        prediction = np.argmax(probs)        
        return prediction, plt.imshow(img)
    

