# 🚗 Vehicle Detection and Traffic Assessment

This project allows users to upload an image of a road (CCTV snapshot), detects vehicles using a YOLO-based deep learning model, and provides a simple traffic assessment (car count and density level). Built with FastAPI, OpenCV, and a user-friendly HTML/CSS/JS frontend.

# 🧠 Features

- 🔍 Object detection on road images using a YOLOv8 model.
- 📸 Image upload interface via a web frontend.
- 💡 Real-time feedback on:
    - Number of cars detected
    - Estimated traffic density (Low / Medium / High)
- ⚙️ FastAPI backend for handling inference
- 🖼️ Clean and modern web UI (HTML/CSS/JavaScript)
- 🔒 CORS enabled and ready for local or production deployment

# 🚀 How to Run
1. Clone the repository
2. Install dependencies : **pip install -r requirements.txt**
3. Start the backend : **uvicorn main:app --reload**
4. Open the frontend

![Screenshot 2025-06-30 151449](https://github.com/user-attachments/assets/ccd2cfb6-e567-409a-ac8c-a6acbc34dc03)
