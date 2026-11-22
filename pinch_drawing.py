
import cv2
import mediapipe as mp
import numpy as np

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Camera could not be opened.")
    raise SystemExit

# ================= HAND TRACKING =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ================= COLORS (BGR) =================
colors = [
    (0, 255, 0),       # 1 Green
    (255, 0, 0),       # 2 Blue
    (0, 0, 255),       # 3 Red
    (0, 165, 255),     # 4 Orange
    (0, 255, 255),     # 5 Yellow
    (255, 255, 0),     # 6 Cyan
    (255, 0, 255),     # 7 Magenta
    (128, 0, 128),     # 8 Purple
    (255, 255, 255),   # 9 White
    (0, 0, 0)          # 0 Black
]

color_names = [
    "Green", "Blue", "Red", "Orange", "Yellow",
    "Cyan", "Magenta", "Purple", "White", "Black"
]

color_index = 0
brush_size = 6
mode = "brush"

# ================= DRAWING =================
strokes = []
redo_stack = []
active_stroke = None

smooth_x = None
smooth_y = None
drawing = False

PINCH_START = 40
PINCH_END = 55

save_number = 1


# ================= RAINBOW =================
def make_rainbow_color(index, total):
    hue = int(index * 179 / max(total - 1, 1))
    hsv = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(int(v) for v in bgr[0, 0])


# ================= RENDER =================
def render_strokes():
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    for stroke in strokes:
        points = stroke["points"]

        if not points:
            continue

        thickness = stroke["size"]
        stroke_mode = stroke["mode"]
        color = stroke["color"]

        if stroke_mode == "eraser":
            draw_color = (0, 0, 0)
        else:
            draw_color = color

        if len(points) == 1:
            cv2.circle(
                canvas, points[0],
                max(thickness // 2, 1),
                draw_color, -1, cv2.LINE_AA
            )

        for i in range(1, len(points)):
            segment_color = draw_color

            if stroke_mode == "rainbow":
                segment_color = make_rainbow_color(i, len(points))

            cv2.line(
                canvas, points[i - 1], points[i],
                segment_color, thickness, cv2.LINE_AA
            )

    if active_stroke is not None:
        points = active_stroke["points"]

        if points:
            thickness = active_stroke["size"]
            color = active_stroke["color"]
            stroke_mode = active_stroke["mode"]

            draw_color = (0, 0, 0) if stroke_mode == "eraser" else color

            if len(points) == 1:
                cv2.circle(
                    canvas, points[0],
                    max(thickness // 2, 1),
                    draw_color, -1, cv2.LINE_AA
                )

            for i in range(1, len(points)):
                segment_color = draw_color

                if stroke_mode == "rainbow":
                    segment_color = make_rainbow_color(i, len(points))

                cv2.line(
                    canvas, points[i - 1], points[i],
                    segment_color, thickness, cv2.LINE_AA
                )

    if any(s["mode"] == "neon" for s in strokes) or (
        active_stroke is not None and active_stroke["mode"] == "neon"
    ):
        glow = cv2.GaussianBlur(canvas, (0, 0), 10)
        canvas = cv2.addWeighted(canvas, 1.0, glow, 0.8, 0)

    return canvas


# ================= STROKE CONTROL =================
def finish_stroke():
    global active_stroke

    if active_stroke is not None:
        if active_stroke["points"]:
            strokes.append(active_stroke)
            redo_stack.clear()

        active_stroke = None


def add_point(x, y):
    global active_stroke

    if active_stroke is None:
        active_stroke = {
            "points": [],
            "color": colors[color_index],
            "size": brush_size,
            "mode": mode
        }

    active_stroke["points"].append((x, y))


# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera frame could not be read.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # ================= HAND DETECTED =================
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame, hand, mp_hands.HAND_CONNECTIONS
        )

        thumb = hand.landmark[4]
        index = hand.landmark[8]

        tx, ty = int(thumb.x * w), int(thumb.y * h)
        ix, iy = int(index.x * w), int(index.y * h)

        distance = np.hypot(tx - ix, ty - iy)

        if smooth_x is None:
            smooth_x, smooth_y = ix, iy
        else:
            smooth_x = int(0.75 * smooth_x + 0.25 * ix)
            smooth_y = int(0.75 * smooth_y + 0.25 * iy)

        if not drawing and distance < PINCH_START:
            drawing = True

        elif drawing and distance > PINCH_END:
            drawing = False
            finish_stroke()

        if drawing:
            add_point(smooth_x, smooth_y)

    else:
        drawing = False
        finish_stroke()
        smooth_x, smooth_y = None, None

    # ================= DISPLAY CANVAS =================
    canvas = render_strokes()

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    drawing_mask = gray > 0
    frame[drawing_mask] = canvas[drawing_mask]

    # ================= CLEAN SCREEN =================
    cv2.imshow("Gesture Paint Studio", frame)

    # ================= KEYBOARD =================
    key = cv2.waitKey(1) & 0xFF

    if ord("1") <= key <= ord("9"):
        finish_stroke()
        color_index = key - ord("1")
        mode = "brush"

    elif key == ord("0"):
        finish_stroke()
        color_index = 9
        mode = "brush"

    elif key == ord("u"):
        finish_stroke()
        if strokes:
            redo_stack.append(strokes.pop())

    elif key == ord("y"):
        finish_stroke()
        if redo_stack:
            strokes.append(redo_stack.pop())

    elif key == ord("e"):
        finish_stroke()
        mode = "eraser"

    elif key == ord("b"):
        finish_stroke()
        mode = "brush"

    elif key == ord("n"):
        finish_stroke()
        mode = "neon"

    elif key == ord("r"):
        finish_stroke()
        mode = "rainbow"

    elif key in (ord("+"), ord("=")):
        brush_size = min(brush_size + 2, 50)

    elif key == ord("-"):
        brush_size = max(brush_size - 2, 2)

    elif key == ord("s"):
        finish_stroke()
        canvas = render_strokes()
        filename = f"drawing_{save_number}.png"
        cv2.imwrite(filename, canvas)
        print(f"Saved: {filename}")
        save_number += 1

    elif key == ord("c"):
        strokes.clear()
        redo_stack.clear()
        active_stroke = None

    elif key == ord("q"):
        break


# ================= CLEANUP =================
cap.release()
hands.close()
cv2.destroyAllWindows()
