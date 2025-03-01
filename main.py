import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from google.colab import files

base_model = MobileNetV2(weights="imagenet")
model = Model(inputs=base_model.input, outputs=base_model.output)

uploaded = files.upload()
for file_name in uploaded.keys():
    test_image_path = file_name

def generate_gradcam(image_path, model):
    """ Generate Grad-CAM heatmap for tampering detection """
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_image = img_to_array(image)
    processed_image = preprocess_input(processed_image)
    processed_image = np.expand_dims(processed_image, axis=0)

    preds = model.predict(processed_image)
    tampering_score = preds.max() * 100
    confidence_score = 100 - tampering_score

    last_conv_layer = model.get_layer("Conv_1")
    grad_model = Model(inputs=model.input, outputs=[last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(processed_image)
        loss = predictions[:, np.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlayed_image = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    return overlayed_image, tampering_score, confidence_score

def detect_tampering(image_path):
    """ Detect tampered regions and display heatmap with scores """
    overlayed_image, tampering_score, confidence_score = generate_gradcam(image_path, model)

    plt.figure(figsize=(8, 6))
    plt.imshow(overlayed_image)
    plt.title(f"🛠️ Tampering Score: {tampering_score:.2f}% | 🔍 Confidence: {confidence_score:.2f}%")
    plt.axis('off')
    plt.show()


detect_tampering(test_image_path)
