from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Colors
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]
color_index = 0
draw_color = colors[color_index]
canvas = None

cap = cv2.VideoCapture(0)

def gen_frames():
    global canvas, draw_color, color_index

    xp, yp = 0, 0
    thumb_open = False

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        if canvas is None:
            canvas = np.zeros_like(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm_list = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    lm_list.append((int(lm.x * w), int(lm.y * h)))

                # Fingers state
                if len(lm_list) != 0:
                    # Thumb: Check open/close
                    if lm_list[4][0] > lm_list[3][0]:
                        if not thumb_open:
                            thumb_open = True
                            color_index = (color_index + 1) % len(colors)
                            draw_color = colors[color_index]
                    else:
                        thumb_open = False

                    # Index finger drawing
                    if lm_list[8][1] < lm_list[6][1] and lm_list[12][1] > lm_list[10][1]:
                        x1, y1 = lm_list[8]
                        if xp == 0 and yp == 0:
                            xp, yp = x1, y1
                        cv2.line(canvas, (xp, yp), (x1, y1), draw_color, 5)
                        xp, yp = x1, y1
                    else:
                        xp, yp = 0, 0

                    # Palm erase
                    palm_size = np.linalg.norm(np.array(lm_list[0]) - np.array(lm_list[9]))
                    if palm_size > 100:
                        canvas = np.zeros_like(frame)

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Merge canvas and frame
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, inv = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, inv)
        frame = cv2.bitwise_or(frame, canvas)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0')
