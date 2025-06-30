from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, cv2, numpy as np, base64
from ultralytics import YOLO

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = YOLO("../yolov8n.pt")

# Create folders
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Function to detect and annotate cars in the image
def annotate_image(image_path):
    image = cv2.imread(image_path)
    results = model.predict(image, verbose=False)[0]
    cars = [box for box in results.boxes if model.names[int(box.cls[0])] == "car"]

    for box in cars:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)

    # Save and encode the result image
    filename = os.path.basename(image_path)
    result_path = os.path.join(RESULT_DIR, f"processed_{filename}")
    cv2.imwrite(result_path, image)

    _, buffer = cv2.imencode('.jpg', image)
    encoded_img = base64.b64encode(buffer).decode('utf-8')

    # Traffic density classification
    density = "Low" if len(cars) < 5 else "Medium" if len(cars) < 15 else "High"
    return len(cars), density, encoded_img

# POST route for image upload and detection
@app.post("/upload")
async def predict_image(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    count, density, encoded_img = annotate_image(filepath)

    return JSONResponse({
        "count": count,
        "density": density,
        "image_url": encoded_img
    })
