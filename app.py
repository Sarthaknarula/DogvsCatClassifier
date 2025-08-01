import gradio as gr
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import load_model

# Load model
model = load_model("dogcatclassifier.keras")

# Prediction function
def predict_image(image):
    image = np.array(image)
    resize = tf.image.resize(image, (256, 256))
    yhat = model.predict(np.expand_dims(resize / 225, 0))
    if yhat > 0.5:
        return f"Predicted class: 🐶 Dog ({yhat[0][0]:.2f})"
    else:
        return f"Predicted class: 🐱 Cat ({1 - yhat[0][0]:.2f})"

# Gradio interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Dog vs Cat Classifier",
    description="Upload an image of a dog or cat to classify."
)

interface.launch()
