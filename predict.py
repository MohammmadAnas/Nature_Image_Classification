import os

from tensorflow import keras
from tensorflow.keras.applications.xception import preprocess_input

from flask import Flask
from flask import request
from flask import jsonify

test_path = 'Intel_Image_Classification_small/seg_test/seg_test'

model = keras.models.load_model('xception2_08_0.9283.h5')

app = Flask('Image_Classification')

@app.route('/predict', methods=['POST'])
def predict():
    
    image_data = request.get_json()

    X = preprocess_input(image_data)
    pred = model.predict(X)

    classes = os.listdir(test_path)
    
   
    result = {
        'Card': dict(zip(classes, pred[0]))
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run()