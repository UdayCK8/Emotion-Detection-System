from flask import Flask, render_template, Response
import cv2
import numpy as np
from tf_keras.models import load_model  # if you installed tf-keras

app = Flask(__name__)

model = load_model('model/emotion_cnn.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

EMOTIONS = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']

COLORS = {
    'Happy':    (0, 220, 100),
    'Sad':      (200, 100, 50),
    'Angry':    (50, 50, 220),
    'Fear':     (180, 80, 220),
    'Surprise': (50, 200, 220)
}

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=(0, -1))

            preds = model.predict(roi, verbose=0)[0]
            emotion = EMOTIONS[np.argmax(preds)]
            confidence = int(np.max(preds) * 100)
            color = COLORS.get(emotion, (255, 255, 255))

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            label = f"{emotion}  {confidence}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y - th - 14), (x + tw + 10, y), color, -1)
            cv2.putText(frame, label, (x + 5, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            bar_x = x + w + 10
            if bar_x + 160 < frame.shape[1]:
                for i, (emo, prob) in enumerate(zip(EMOTIONS, preds)):
                    bar_y = y + i * 24
                    bar_len = int(prob * 140)
                    emo_color = COLORS.get(emo, (200, 200, 200))
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_len, bar_y + 16), emo_color, -1)
                    cv2.putText(frame, f"{emo[:3]} {int(prob*100)}%",
                                (bar_x + bar_len + 4, bar_y + 13),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)