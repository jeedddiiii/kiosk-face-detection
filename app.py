import os
import cv2
from flask import Flask, Response, render_template, jsonify
from deepface import DeepFace
import numpy as np
import datetime
import base64
import io


# Initialize the Flask app
app = Flask(__name__)

# Face detection cascade classifier path (adjust if needed)
face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

emotion_model = DeepFace.build_model('Emotion')

backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']
models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib', 'ArcFace', 'SFace', 'Facenet512']
metrics = ['cosine', 'euclidean', 'euclidean_l2']


current_name = "Unknown"
current_emotion = "Unknown"
current_date_time = ""
current_text = ""
current_image = None
current_face = None

def analyze_emotion(face_crop):
    try:
        # Detect emotions in the face
        emotions = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
        return emotions[0]['dominant_emotion']
    except Exception as e:
        print("An error occurred in emotion analysis: ", e)
        return "Unknown"

def recognize_face(face_crop):
    try:
        people = DeepFace.find(img_path=face_crop, db_path="img/", model_name=models[0], distance_metric=metrics[0], enforce_detection=False)
        return people
    except Exception as e:
        print("Error in face recognition:", str(e))
        return []


def generate_frames():
    global current_name, current_emotion, current_date_time, current_text, current_image, current_face

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

            
            face_crop = np.copy(frame[y:y+h, x:x+w])

            # Analyze emotion
            current_emotion = analyze_emotion(face_crop)

            # Recognize face
            people = recognize_face(face_crop)
           

            for person in people:
                if not person['source_x'].empty:
                    x = person['source_x'][0]
                    y = person['source_y'][0]
                    w = person['source_w'][0]
                    h = person['source_h'][0]

                    try:
                        distance = person['target_x'][0]
                        print("Distance:", distance)  # Add this line for debugging

                        if distance > 1:
                            current_name = person['identity'][0].split('/')[1]
                        else:
                            current_name = "Unknown"
                        cv2.putText(frame, current_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    except KeyError:
                        current_name = "Unknown"
                        print("Distance not found")
                        cv2.putText(frame, current_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            
            is_success, buffer = cv2.imencode(".jpg", frame)
            if is_success:
                io_buf = io.BytesIO(buffer)
                current_image = base64.b64encode(io_buf.getvalue()).decode('utf-8')

            is_success, buffer = cv2.imencode(".jpg", face_crop)
            if is_success:
                io_buf = io.BytesIO(buffer)
                current_face = base64.b64encode(io_buf.getvalue()).decode('utf-8')

            
            current_date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if current_emotion == "happy":
                current_text = "ยิ้มหาพ่อเธอหรือ"
            elif current_emotion == "angry":
                current_text = "หน้าบึ้งหาพ่อเธอหรือ"
            elif current_emotion == "fear":
                current_text = "กลัวหาพ่อเธอหรือ"
            elif current_emotion == "sad":
                current_text = "เศร้าหาพ่อเธอหรือ"
            elif current_emotion == "surprise":
                current_text = "แปลกหาพ่อเธอหรือ"
            elif current_emotion == "neutral":
                current_text = "เฉยๆหาพ่อเธอหรือ"
            elif current_emotion == "disgust":
                current_text = "เกลียดหาพ่อเธอหรือ"

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
    return jsonify(name=current_name, emotion=current_emotion, date_time=current_date_time, text=current_text, image=current_image, face=current_face)



if __name__ == '__main__':
    # Run the Flask server
    app.run(host='0.0.0.0', debug=True)
