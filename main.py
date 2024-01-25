import os
import cv2
import base64
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ColorExtractor import ColorExtractor

app = FastAPI()
cer = ColorExtractor()

@app.post("/upload")
async def create_upload_file(file: UploadFile = File(...)):
    file_path = f"{MEDIA_ROOT}/{file.filename}"

    img = np.frombuffer(file.file.read(), dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    fp_png = f'{file_path.split(".")[0]}.png'
    cv2.imwrite(fp_png, img)

    pred = cer.predict(img)
    mask = cer.getMask(img, pred)

    if mask is None:
        return {"filename": file.filename, "file_path": file_path, "colors": {}}

    cmap = cer.extractFromImageMaskPair(img, mask)

    thumbnail = cer.getThumbnail(img, mask, pred)
    _, thumbnail_buffer = cv2.imencode('.jpg', thumbnail)
    thumbnail_bytes = base64.b64encode(thumbnail_buffer)

    return {"filename": file.filename, "file_path": file_path, "colors": cmap, "imageBase64": thumbnail_bytes}

MEDIA_ROOT = 'uploads'
os.makedirs(MEDIA_ROOT, exist_ok=True)

# prevent CORS error
origins = ["*"]
origins = [
    "http://localhost",
    "http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
