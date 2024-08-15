from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('dog_breed_model.h5')


# Assuming you have the class indices saved as part of the model
class_indices = model.class_indices  # This only works if you saved class indices as part of the model

# Get the labels from class indices
if class_indices:
    class_labels = {v: k for k, v in class_indices.items()}
    print(class_labels)
else:
    print("Class indices not found in the model.")


# Class labels
class_labels = list(np.load('class_labels.npy'))

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    img_path = f"temp/{file.filename}"
    file.save(img_path)

    img = prepare_image(img_path)
    pred = model.predict(img)
    pred_class = np.argmax(pred, axis=1)[0]
    label = class_labels[pred_class]

    return jsonify({"breed": label})

if __name__ == '__main__':
    app.run(debug=True)
