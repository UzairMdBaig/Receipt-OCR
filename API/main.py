from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
from scipy.special import softmax
import onnxruntime as ort
from PIL import Image
import numpy as np
import math
import cv2
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        
    allow_credentials=True,
    allow_methods=["*"],          
    allow_headers=["*"],          
)

det_model = ort.InferenceSession("./models/en_PP-OCRv3_det_infer.onnx")
rec_model = ort.InferenceSession("./models/en_PP-OCRv3_rec_infer.onnx")


mean_det = [0.485, 0.456, 0.406] 
std_det = [0.229, 0.224, 0.225]

with open("en_dict.txt", "r", encoding="utf-8") as f:
    label_map = [line.strip() for line in f if line.strip()]

def detector_preprocess(image):
    #Adjusitng image such that the longer side is 960 if it exceeds 960 while keeping the same aspect ratio
    limit_side_len = 960
    h, w = image.shape[:2]
    ratio = 1.0
    
    if max(h, w) > limit_side_len:
        ratio = limit_side_len / max(h, w)
        new_w = max(1, int(w * ratio))
        new_h = max(1, int(h * ratio))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (new_w, new_h))

    # Applying padding so each side is a multipl of 32
    h, w = image.shape[:2]
    pad_h = int(math.ceil(h / 32) * 32)
    pad_w = int(math.ceil(w / 32) * 32)
    padded = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
    padded[:h, :w, :] = image
    image = padded
    original_image = image.copy()
    image = image.astype(np.float32) / 255.0
    image = (image - mean_det) / std_det
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image, original_image

def recognizer_preprocess(cropped):
    # Resize height to 48, keep aspect ratio
    h, w = cropped.shape[:2]
    new_h = 48
    new_w = max(1, int(w * (new_h / h)))
    cropped = cv2.resize(cropped, (new_w, new_h))

    # Normalize
    cropped = cropped.astype(np.float32) / 255.0
    mean_rec = [0.5, 0.5, 0.5]
    std_rec = [0.5, 0.5, 0.5]
    cropped = (cropped - mean_rec) / std_rec

    # Channels-first
    cropped = np.transpose(cropped, (2, 0, 1))

    # Add batch dimension
    cropped = np.expand_dims(cropped, axis=0)

    return cropped


def ctc_decode(pred_indices, label_map, blank=0):
    results = []
    n_labels = len(label_map)

    for batch in pred_indices:
        prev = blank
        text = []
        for idx in batch:
            if idx == prev or idx == blank:
                prev = idx
                continue
            if idx < n_labels:
                text.append(label_map[idx-1])
            prev = idx
        results.append("".join(text))
    return results

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        file_bytes = file.file.read()

        if file.content_type == "application/pdf":
            pillow_images_list = convert_from_bytes(file_bytes)
            images = [np.array(img) for img in pillow_images_list]
        elif file.content_type.startswith("image/"):
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            images = [np.array(img)]
        else:
            raise HTTPException(status_code=400, detail="Only PDFs or images are allowed.")

        text_results = []

        for image in images:
            # Detector preprocessing
            det_input, original_image = detector_preprocess(image)
            det_input = det_input.astype(np.float32)
            det_outputs = det_model.run(None, {"x": det_input})
            det_output = det_outputs[0][0, 0, :, :]

            threshold = det_output > 0.4
            contours, _ = cv2.findContours(
                threshold.astype(np.uint8),
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE
            )

            boxes = []
            crops = []

            for contour in contours:
                if cv2.contourArea(contour) < 100:
                    continue
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                boxes.append(box)

                # Crop straightened region
                pts_src = np.array(box, dtype="float32")
                pad = 10  

                # Compute width and height of the box
                w_box = max(1, int(np.linalg.norm(pts_src[0] - pts_src[1])) + 2*pad)
                h_box = max(1, int(np.linalg.norm(pts_src[0] - pts_src[3])) + 2*pad)

                # Compute the center of the box
                center = pts_src.mean(axis=0)

                # Shift destination points so padding is added around the text
                pts_dst = np.array([
                    [0, 0],
                    [w_box-1, 0],
                    [w_box-1, h_box-1],
                    [0, h_box-1]
                ], dtype="float32")

                # Move source points outward to add padding
                direction = pts_src - center  # vector from center
                pts_src_padded = center + direction * ((np.array([w_box, h_box]) / np.array([w_box-2*pad, h_box-2*pad])).reshape(1,2))
                M = cv2.getPerspectiveTransform(pts_src_padded.astype(np.float32), pts_dst)
                cropped = cv2.warpPerspective(original_image, M, (w_box, h_box))
                crops.append(cropped)

            if len(crops) == 0:
                continue

            # ----------------- Recognizer preprocessing in batch -----------------
            rec_inputs = []
            for cropped in crops:
                h, w = cropped.shape[:2]
                if h > w * 1.5:
                    continue
                rec_input = recognizer_preprocess(cropped)
                rec_inputs.append(rec_input)

            # Stack along batch dimension
            max_w = max([crop.shape[3] for crop in rec_inputs])  # crop.shape[3] is width
            rec_inputs_padded = []

            for crop in rec_inputs:
                b, c, h, w = crop.shape
                padded = np.zeros((b, c, h, max_w), dtype=np.float32)
                padded[:, :, :, :w] = crop
                rec_inputs_padded.append(padded)

            rec_inputs = np.concatenate(rec_inputs_padded, axis=0)
            rec_inputs = rec_inputs.astype(np.float32)
            rec_outputs = rec_model.run(None, {"x": rec_inputs})[0]
            probs = softmax(rec_outputs, axis=2)
            pred_indices = probs.argmax(axis=2)
            texts = ctc_decode(pred_indices, label_map)
            text_results.extend(texts)

        return {"texts": text_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))