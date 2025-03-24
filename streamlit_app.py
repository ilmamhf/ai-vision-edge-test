from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
import io
from fastapi import FastAPI, File, UploadFile
from threading import Thread
import uvicorn

# ====== Load Model dan Label ======
np.set_printoptions(suppress=True)
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# ====== FastAPI untuk API Prediksi ======
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Preprocessing gambar
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Input ke model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Prediksi
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = float(prediction[0][index])

    return {"class": class_name, "confidence": confidence_score}

# ====== Menjalankan FastAPI di Thread Terpisah ======
def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8000)

thread = Thread(target=run_api)
thread.daemon = True
thread.start()

# ====== UI Streamlit ======
st.title("Cloud AI Vision API")
st.write("Gunakan endpoint `/predict` untuk mengirim gambar dan mendapatkan hasil deteksi.")