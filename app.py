import os
import cv2
from flask import Flask, Response, render_template, jsonify
from deepface import DeepFace
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Face detection cascade classifier path (adjust if needed)
face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

emotion_model = DeepFace.build_model('Emotion')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
print(recognizer)

# Initialize user IDs and associated names
id = 0
# Don't forget to add names associated with user IDs
names = ['None', 'Pang','Jedi', 'Aomsun', 'pa', 'Poon']

current_name = "Unknown"
current_emotion = "Unknown"


def generate_frames():
    global current_name, current_emotion

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
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            
            face_crop = np.copy(frame[y:y+h, x:x+w])
            try:
                # Detect emotions in the face
                emotions = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
                current_emotion = emotions[0]['dominant_emotion']


                cv2.putText(frame, emotions[0]['dominant_emotion'], (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)        


                # Display the emotions on the frame
            except Exception as e:
                print("An error occurred: ", e)

            # Proba greater than 40
            if confidence > 40:
                try:
                    # Recognized face
                    name = names[id]
                    current_name = names[id]

                    confidence = "  {0}%".format(round(confidence))
                except IndexError as e:
                    name = "Who?"
                    confidence = "N/A"
            else:
                # Unknown face
                name = "Who?"
                confidence = "N/A"

            cv2.putText(frame, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, confidence, (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

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

@app.route('/data')
def data():
    # Return the current name and emotion as JSON
    return jsonify(name=current_name, emotion=current_emotion)



if __name__ == '__main__':
    # Run the Flask server
    app.run(host='0.0.0.0', debug=True)
