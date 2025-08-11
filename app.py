from flask import Flask, render_template, Response
from painter import VirtualPainter

app = Flask(__name__)
painter = VirtualPainter()

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    for frame in painter.run():
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
