# 🧾 Receipt / PDF OCR with ONNX (Images + PDFs)  

This project demonstrates an end-to-end OCR (Optical Character Recognition) pipeline that extracts text from receipts, invoices, and PDFs using **ONNX models**.  

**Models:**  
- Detection model (`en_PP-OCRv3_det_infer.onnx`) for locating text regions  
- Recognition model (`en_PP-OCRv3_rec_infer.onnx`) for decoding cropped text
 
**Backend:** FastAPI serving the ONNX inference with `onnxruntime`, `OpenCV`, and `pdf2image`  
**Frontend:** React app to upload and visualize extracted text from receipts or PDF files  
**Deployment:** Fully containerized with Docker and orchestrated using Docker Compose (Nginx reverse proxy for API + frontend)  

---

## 📂 Project Structure  
```plaintext
.
├── API/                      # FastAPI backend
│   ├── models/               # OCR ONNX models
│   │   ├── en_PP-OCRv3_det_infer.onnx
│   │   └── en_PP-OCRv3_rec_infer.onnx
│   ├── en_dict.txt           # Dictionary for text decoding
│   ├── Dockerfile
│   ├── main.py               # OCR API code
│   └── requirements.txt
│
├── frontend/                 # React frontend
│   ├── Dockerfile
│   ├── nginx.conf            # Reverse proxy config
│   ├── src/
│   │   └── App.jsx           # Main UI logic
│   └── package.json
│
├── docker-compose.yml        # Multi-service setup
│
└── README.md
```

---

## 📖 Model & Preprocessing
**🔍 Detection Preprocessing**

- Resize input image so longest side ≤ 960 px (keeping aspect ratio)
- Pad image so height/width are multiples of 32
- Normalize using mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
- Output: text region bounding boxes


**🔡 Recognition Preprocessing**
- Cropped text regions are resized to height = 48 (maintaining aspect ratio)
- Normalize using mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]
- Channels-first, batched for ONNX inference
- Decoded with CTC decode to output readable text

  ---

## ▶️ Running the Project

**1. Clone the repository**
```bash
git clone https://github.com/your-username/Receipt-OCR.git
cd Receipt-OCR
```

**2. Build and run with Docker Compose**
```bash
docker-compose up --build
```

**3. Access the frontend**
```bash
👉 http://localhost:100
```

---


## 🌐 API Endpoint

**POST /predict**

- **Input**: Single PDF file (application/pdf)/ Single image file (image/jpeg, image/png, etc.)
- **Output**: JSON containing detected text strings
```json
{
  "texts": ["Subtotal 20.00", "Tax 1.50", "Total 21.50"]
}
```

---

## 🛠 Tech Stack

- **Backend**: FastAPI, ONNX Runtime, OpenCV, NumPy, pdf2image, SciPy
- **Frontend**: React, TailwindCSS
- **Frontend Proxy**: Nginx
- **Deployment**: Docker + Docker Compose
