from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(_name_)
model = load_model('hematovision_model.h5')
class_names = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img = request.files['image']
        if img:
            img_path = os.path.join('static', img.filename)
            img.save(img_path)

            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = image / 255.0

            prediction = model.predict(image)
            class_idx = np.argmax(prediction)
            label = class_names[class_idx]

            return render_template('index.html', label=label, img_path=img_path)

    return render_template('index.html', label=None)

if _name_ == '_main_':
    app.run(debug=True)