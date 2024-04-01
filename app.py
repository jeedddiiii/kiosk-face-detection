import os
import cv2
from flask import Flask, Response, render_template, jsonify, request
from deepface import DeepFace
import numpy as np
import datetime
import base64
import io
import pyttsx3
import psycopg2
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import threading
import queue
import requests
from werkzeug.utils import secure_filename

TH_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_THAI"
# Initialize the Flask app
app = Flask(__name__)

# Face detection cascade classifier path (adjust if needed)
face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

emotion_model = DeepFace.build_model('Emotion')

backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']
models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib', 'ArcFace', 'SFace', 'Facenet512']
metrics = ['cosine', 'euclidean', 'euclidean_l2']

current_name = ""
current_emotion = "Unknown"
current_date_time = ""
current_text = ""
current_image = None
current_face = None

face_queue = queue.Queue()
emotion_queue = queue.Queue()
def send_data(current_name, current_date_time, current_emotion, source_id, current_face, current_image):
    # The data to be inserted
    data = {
        'name': current_name,
        'date_time': current_date_time,
        'emotion': current_emotion,
        'source_id': source_id,
        'face_img': current_face,
        'environment_img': current_image
    }

    # Send the POST request
    response = requests.post('http://localhost:8080/insert-transaction', json=data)

    # Print the response
    print(response.json())

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

previous_name = "Unknown"
last_face_detection_time = datetime.datetime.now()

def recognize_face_thread():
    global current_name, last_face_detection_time

    while True:
        face_crop = face_queue.get()
        # ตรวจสอบเวลาที่ตรวจจับหน้าล่าสุด
        current_time = datetime.datetime.now()
        time_difference = current_time - last_face_detection_time
        if time_difference.total_seconds() < 3:
            # ถ้ายังไม่ผ่านเวลาสามวินาทีให้ข้ามการประมวลผล
            face_queue.task_done()
            continue 

        people = recognize_face(face_crop)
        if people:
            current_name = people[0]['identity'][0].split('/')[1]
        else:
            current_name = "Unknown"

        # อัปเดตเวลาที่ตรวจจับหน้าล่าสุด
        last_face_detection_time = datetime.datetime.now()
        
        emotion_queue.put((current_name, face_crop))
        face_queue.task_done()

def analyze_emotion_thread():
    global current_emotion, current_date_time, current_text, current_image, current_face, previous_name
    while True:
        current_name, face_crop = emotion_queue.get()
    
        current_emotion = analyze_emotion(face_crop)

        if current_name != previous_name:
            is_success, buffer = cv2.imencode(".jpg", frame)
            if is_success:
                io_buf = io.BytesIO(buffer)
                current_image = base64.b64encode(io_buf.getvalue()).decode('utf-8')
            # Convert face crop image to base64
            is_success, buffer = cv2.imencode(".jpg", face_crop)
            if is_success:
                io_buf = io.BytesIO(buffer)
                current_face = base64.b64encode(io_buf.getvalue()).decode('utf-8')
                
            # Update current date and time
            current_date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if current_emotion == "happy":
                current_text = "ดูท่าคุณจะดีใจมากเลยนะ"
            elif current_emotion == "angry":
                current_text = "คุณดูโกรธมากเลย มีอะไรให้ช่วยบ้างไหม"
            elif current_emotion == "fear":
                current_text = "อย่ากลัวนะ ไม่มีอะไรน่ากลัว"
            elif current_emotion == "sad":
                current_text = "คุณดูเศร้ามาก มีอะไรให้ช่วยปลอบใจบ้างไหม"
            elif current_emotion == "surprise":
                current_text = "ว้าว! คุณดูตกใจมาก"
            elif current_emotion == "neutral":
                current_text = "สบายดีนะ"
            elif current_emotion == "disgust":
                current_text = "คุณดูรังเกียจบางอย่างเหรอ"

            jarvis(current_text)
            send_data(current_name, current_date_time, current_emotion, 1, current_face, current_image)
            
            previous_name = current_name
        else:
            continue

        emotion_queue.task_done()

def jarvis(current_text):
    engine = pyttsx3.init()
    engine.setProperty('voice', TH_voice_id)
    engine.say(current_text)
    engine.runAndWait()

def generate_frames():
    global frame, previous_name

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    face_thread = threading.Thread(target=recognize_face_thread, daemon=True)
    emotion_thread = threading.Thread(target=analyze_emotion_thread, daemon=True)
    face_thread.start()
    emotion_thread.start()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, 90) 

        if not ret:
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
        # If no faces detected, set previous_name to "unknown"
            previous_name = "unknown"
            
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            face_crop = np.copy(frame[y:y+h, x:x+w])
            face_queue.put(face_crop)

        # Encode frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame with rectangles drawn
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    face_queue.join()
    emotion_queue.join()

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
    return jsonify(name=current_name, date_time=current_date_time, text=current_text, face=current_face)

@app.route('/delete_representations', methods=['POST'])
def delete_representations():
    try:
        os.remove('img/representations_vgg_face.pkl')
        return 'File deleted.'
    except Exception as e:
        return f'An error occurred: {str(e)}', 500

@app.route('/store_image', methods=['POST'])
def store_image():
    name = request.form['name']
    image = request.form['image']

    # Create a directory with the name if it doesn't exist
    if not os.path.exists('img/' + name):
        os.makedirs('img/' + name)

    # Decode the base64 image
    image_data = base64.b64decode(image)
    image_filename = secure_filename('image.jpg')

    # Save the image in the directory
    with open(os.path.join('img', name, image_filename), 'wb') as f:
        f.write(image_data)

    return 'Image stored.', 200
if __name__ == '__main__':
    # Run the Flask server
    app.run(host='0.0.0.0', debug=True)