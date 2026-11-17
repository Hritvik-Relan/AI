import numpy as np
from flask import Flask, request, jsonify, render_template
import time
import os
import tensorflow as tf
from PIL import Image, ImageOps
app = Flask(__name__)
model = tf.keras.models.load_model('keras_model.h5')
with open('labels.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]
imgpath = 'draw.png'
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    imgfile = request.files['image']
    
    imgfile.save(imgpath)
    img = Image.open(imgpath).convert('RGB')
    size = (224, 224)
    img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    imgarray = np.asarray(img)
    normimgarray = (imgarray.astype(np.float32) / 127.5) -1
    dataarr = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    dataarr[0] = normimgarray
    prediction = model.predict(dataarr)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence = float(prediction[0][index]*100)
    if os.path.exists(imgpath):
        os.remove(imgpath)
    return jsonify({'class_name': class_name, 'confidence': confidence})    
if __name__ == '__main__':
    app.run(debug=True)