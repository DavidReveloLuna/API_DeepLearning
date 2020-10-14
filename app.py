from __future__ import division, print_function

# Keras
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input

# Flask 
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

import os
import numpy as np
import cv2


width_shape = 224
height_shape = 224

names = ['AFRICAN FIREFINCH','ALBATROSS','ALEXANDRINE PARAKEET','AMERICAN AVOCET','AMERICAN BITTERN',
         'AMERICAN COOT','AMERICAN GOLDFINCH','AMERICAN KESTREL','AMERICAN PIPIT','AMERICAN REDSTART']

# Definimos una instancia de Flask
app = Flask(__name__)

# Path del modelo preentrenado
MODEL_PATH = 'models/model_RS50.h5'

# Cargamos el modelo preentrenado
model = load_model(MODEL_PATH)

print('Modelo cargado exitosamente. Verificar http://127.0.0.1:5000/')

# Realizamos la predicción usando la imagen cargada y el modelo
def model_predict(img_path, model):

    img=cv2.resize(cv2.imread(img_path), (width_shape, height_shape), interpolation = cv2.INTER_AREA)
    x=np.asarray(img)
    x=preprocess_input(x)
    x = np.expand_dims(x,axis=0)
    
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Página principal
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Obtiene el archivo del request
        f = request.files['file']

        # Graba el archivo en ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Predicción
        preds = model_predict(file_path, model)

        print('PREDICCIÓN', names[np.argmax(preds)])
        
        # Enviamos el resultado de la predicción
        result = str(names[np.argmax(preds)])              
        return result
    return None


if __name__ == '__main__':
    app.run(debug=False, threaded=False)

