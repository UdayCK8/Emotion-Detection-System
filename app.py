from flask import Flask, render_template, Response, request
import cv2
import os
import webbrowser
from threading import Timer
from deepface import DeepFace

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

camera = cv2.VideoCapture(0)

# ----------------------------------------------------
# Auto Open Browser
# ----------------------------------------------------

def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

# ----------------------------------------------------
# Webcam Emotion Detection
# ----------------------------------------------------

def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break

        try:

            results = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False
            )

            if isinstance(results, list):
                results = results[0]

            emotion = results["dominant_emotion"]

            cv2.putText(
                frame,
                f"Emotion: {emotion}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        except Exception as e:
            print(e)

        ret, buffer = cv2.imencode(".jpg", frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

# ----------------------------------------------------
# Home Page
# ----------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

# ----------------------------------------------------
# Webcam Feed
# ----------------------------------------------------

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# ----------------------------------------------------
# Upload Image Detection
# ----------------------------------------------------

@app.route("/upload", methods=["GET", "POST"])
def upload():

    emotion = None

    if request.method == "POST":

        if "image" not in request.files:
            return render_template(
                "upload.html",
                emotion="No file selected"
            )

        file = request.files["image"]

        if file.filename == "":
            return render_template(
                "upload.html",
                emotion="No file selected"
            )

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            file.filename
        )

        file.save(filepath)

        try:

            result = DeepFace.analyze(
                img_path=filepath,
                actions=["emotion"],
                enforce_detection=False
            )

            if isinstance(result, list):
                result = result[0]

            emotion = result["dominant_emotion"]

        except Exception as e:
            emotion = str(e)

    return render_template(
        "upload.html",
        emotion=emotion
    )

# ----------------------------------------------------
# Health Check
# ----------------------------------------------------

@app.route("/health")
def health():
    return {
        "status": "running",
        "camera": camera.isOpened()
    }

# ----------------------------------------------------
# Main
# ----------------------------------------------------

if __name__ == "__main__":

    Timer(2, open_browser).start()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )