import os
import cv2
from flask import Flask, Response, render_template

# Initialize the Flask app
app = Flask(__name__)

# Face detection cascade classifier path (adjust if needed)
face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')


def generate_frames():
    # Initialize video capture
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Ensure we have the face cascade classifier downloaded
        if not os.path.exists(face_cascade_path):
            print("Downloading face cascade classifier...")
            cv2.dnn_fetchNetFromURL(
                'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml',
                face_cascade_path,
                cv2.dnn.FetchNetFromURL_METHOD_HTTP)
            print("Download complete.")

        # Load the face cascade classifier
        face_cascade = cv2.CascadeClassifier(face_cascade_path)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # Encode frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame with rectangles drawn
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/')
def index():
    # Render the HTML template with the video source
    return render_template('index.html')


@app.route('/video')
def video():
    # Video streaming route
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # Run the Flask server
    app.run(host='0.0.0.0', debug=True)
