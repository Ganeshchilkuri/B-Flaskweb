from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub

# Define the custom objects dictionary
custom_objects = {
    'TextVectorization': tf.keras.layers.TextVectorization,
    'StringLookup': tf.keras.layers.StringLookup,
    'Embedding': tf.keras.layers.Embedding,
    'LSTM': tf.keras.layers.LSTM,
    'LSTMCell': tf.keras.layers.LSTMCell,
    'Dense': tf.keras.layers.Dense,
    'Dropout': tf.keras.layers.Dropout,
    'KerasLayer': hub.KerasLayer  # If you're using TensorFlow Hub layers
}

# Load the model with custom objects
model = tf.keras.models.load_model("wines_review_model.h5", custom_objects=custom_objects)

# Assuming you have the vocabulary list used during the training
vocab_list = ['', '[UNK]', 'and', 'the', 'a', 'of', 'with', 'this', 'is', 'wine',
       'flavors', 'in', 'it', 'to', 'its', 'on', 'fruit', 'aromas',
       'palate', 'that']

# Find the TextVectorization and StringLookup layers in the model
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.TextVectorization):
        layer.set_vocabulary(vocab_list)
    if isinstance(layer, tf.keras.layers.StringLookup):
        layer.set_vocabulary(vocab_list)

# Preprocess the input text for prediction
def preprocess_text(text):
    return tf.data.Dataset.from_tensor_slices([text]).batch(1)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    preprocessed_text = preprocess_text(text)
    prediction = model.predict(preprocessed_text)
    # Assuming prediction is a probability or score
    result = "Positive" if prediction >= 0.5 else "Negative"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
