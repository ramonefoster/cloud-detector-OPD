from tensorflow.python.keras.models import load_model
from PIL import Image, ImageOps  
import numpy as np
import requests
from io import BytesIO
import time

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("converted_savedmodel/model.savedmodel", compile=False)
class_names = open("converted_savedmodel/labels.txt", "r").readlines()

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

while True:
    # Replace this with the URL to your image
    try:
        url = "https://coopd.lna.br:8090/img/allsky340c.jpg"
        response = requests.get(url, verify=False)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except:
        time.sleep(30)
        continue

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)

    time.sleep(120)
