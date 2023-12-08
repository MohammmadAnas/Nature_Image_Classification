#import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor


interpreter = tflite.Interpreter(model_path='intel-classification-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

preprocessor = create_preprocessor('xception', target_size=(299, 299))


#url = 'https://img.freepik.com/free-photo/winter-hiking-adventure-majestic-frozen-cliff-generative-ai_188544-12613.jpg?w=1380&t=st=1695997351~exp=1695997951~hmac=4f0ab7ac7bd59227e0f2c5ba977b35eeed35c52a8260d1efd89b4de734b8a8ac'

classes = [
   'buildings',
    'forest',
    'glacier',
    'mountain',
    'sea',
    'street'
]

def predict(url):
    X = preprocessor.from_url(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()

    return dict(zip(classes, float_predictions))

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result
