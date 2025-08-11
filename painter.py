import cv2
import numpy as np
import mediapipe as mp

class VirtualPainter:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.canvas = None
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                       (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        self.color_index = 0
        self.draw_color = self.colors[self.color_index]
        self.xp, self.yp = 0, 0
        self.prev_thumb_state = None
        self.cap = cv2.VideoCapture(0)

    def run(self):
        while True:
            success, img = self.cap.read()
            if not success:
                break
            img = cv2.flip(img, 1)

            if self.canvas is None:
                self.canvas = np.zeros_like(img)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    lm_list = []
                    h, w, _ = img.shape
                    for lm in hand_landmarks.landmark:
                        lm_list.append((int(lm.x * w), int(lm.y * h)))

                    thumb_open = lm_list[4][0] > lm_list[3][0]
                    index_open = lm_list[8][1] < lm_list[6][1]
                    middle_open = lm_list[12][1] < lm_list[10][1]
                    ring_open = lm_list[16][1] < lm_list[14][1]
                    pinky_open = lm_list[20][1] < lm_list[18][1]

                    # Color change
                    if self.prev_thumb_state is None:
                        self.prev_thumb_state = thumb_open
                    elif thumb_open and not self.prev_thumb_state:
                        self.color_index = (self.color_index + 1) % len(self.colors)
                        self.draw_color = self.colors[self.color_index]
                    self.prev_thumb_state = thumb_open

                    # Draw
                    if index_open and not middle_open and not ring_open and not pinky_open:
                        x1, y1 = lm_list[8]
                        if self.xp == 0 and self.yp == 0:
                            self.xp, self.yp = x1, y1
                        cv2.line(self.canvas, (self.xp, self.yp), (x1, y1), self.draw_color, 5)
                        self.xp, self.yp = x1, y1
                    else:
                        self.xp, self.yp = 0, 0

                    # Erase
                    if all([thumb_open, index_open, middle_open, ring_open, pinky_open]):
                        self.canvas = np.zeros_like(img)

                    self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, inv_canvas = cv2.threshold(gray_canvas, 50, 255, cv2.THRESH_BINARY_INV)
            inv_canvas = cv2.cvtColor(inv_canvas, cv2.COLOR_GRAY2BGR)
            img = cv2.bitwise_and(img, inv_canvas)
            img = cv2.bitwise_or(img, self.canvas)

            # Show current color
            cv2.rectangle(img, (0, 0), (100, 100), self.draw_color, -1)

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield frame
