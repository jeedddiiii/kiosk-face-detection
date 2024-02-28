from flask import Flask, Response, render_template
import cv2

# Initialize the Flask app
app = Flask(__name__)


def generate_frames():
  # Initialize video capture
  cap = cv2.VideoCapture(0)

  while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
      break

    # Encode frame in JPEG format
    ret, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()

    # Yield the frame
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
