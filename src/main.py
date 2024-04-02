from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import numpy as np
from pathlib import Path
import os.path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = FastAPI()

project_dir = Path(__file__).resolve().parents[1]
cookie_recognition_model_path = os.path.join(project_dir, "cookie_recognition.h5")

labels = {
    0: "animalito",
    1: "chokis",
    2: "cremax-chocolate",
    3: "emperador-combinado",
    4: "maravillas",
    5: "marianitas",
    6: "marias",
    7: "oreo",
}
model = tf.keras.models.load_model(cookie_recognition_model_path)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/cookie-classifier")
async def upload_image(image: UploadFile = File(...)):
    # Leer imagen como bytes
    image_bytes = await image.read()

    # Decodificar bytes a un array de NumPy
    image_np = np.frombuffer(image_bytes, np.uint8)

    # Convertir a formato BGR para OpenCV
    image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # redimensionar imagen a 224,224,3
    image_bgr = cv2.resize(image_bgr, (224, 224))

    return {"cookie_type": output(image_bgr)}


def output(img):
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    return res
