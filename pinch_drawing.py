
import colorsys
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Camera could not be opened.")
    raise SystemExit

# ================= HAND TRACKING =================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ================= COLORS =================
colors = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (0, 165, 255),
    (0, 255, 255),
    (255, 255, 0),
    (255, 0, 255),
    (128, 0, 128),
    (255, 255, 255),
    (1, 1, 1)
]

color_names = [
    "Green", "Blue", "Red", "Orange", "Yellow",
    "Cyan", "Magenta", "Purple", "White", "Black"
]

color_index = 0
brush_thickness = 6
mode = "brush"

# ================= DRAWING DATA =================
strokes = []
redo_strokes = []
active_stroke = None

pinch_start_threshold = 40
pinch_end_threshold = 55

smoothed_point = None
is_drawing = False
drawing_paused = False

canvas = None
glow_layer = None

clear_armed_until = 0.0

# ================= TOAST NOTIFICATIONS =================
toast_message = ""
toast_until = 0.0


def show_toast(message, duration=2.0):
    global toast_message, toast_until

    toast_message = message
    toast_until = time.monotonic() + duration


# ================= UNIQUE FILE NAMES =================
def unique_filename(prefix):
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    filename = f"{prefix}_{timestamp}.png"
    counter = 1

    while os.path.exists(filename):
        filename = f"{prefix}_{timestamp}_{counter}.png"
        counter += 1

    return filename


# ================= TOP UI =================
def draw_ui(frame):
    h, w, _ = frame.shape
    panel_height = min(92, h)

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (w, panel_height),
        (20, 20, 20),
        -1
    )

    cv2.addWeighted(
        overlay, 0.75,
        frame, 0.25,
        0, frame
    )

    cv2.putText(
        frame,
        "Brush: B   Eraser: E   Neon: N   Rainbow: R   Size: + / -",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (235, 235, 235),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "Undo: U   Redo: Y   Clear: C   Save: S   Camera: F   Pause: P",
        (12, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.37,
        (235, 235, 235),
        1,
        cv2.LINE_AA
    )

    status = (
        f"Color: {color_names[color_index]}    "
        f"Size: {brush_thickness}    "
        f"Mode: {mode.capitalize()}"
    )

    if drawing_paused:
        status += "    PAUSED"

    cv2.putText(
        frame,
        status,
        (12, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # Toast notification
    if toast_message and time.monotonic() < toast_until:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1

        text_size = cv2.getTextSize(
            toast_message,
            font,
            font_scale,
            thickness
        )[0]

        box_width = text_size[0] + 30
        box_height = 38

        x = max(5, w - box_width - 12)
        y = min(105, h - box_height - 5)

        cv2.rectangle(
            frame,
            (x, y),
            (x + box_width, y + box_height),
            (35, 35, 35),
            -1
        )

        cv2.rectangle(
            frame,
            (x, y),
            (x + box_width, y + box_height),
            (80, 200, 120),
            1
        )

        cv2.putText(
            frame,
            toast_message,
            (x + 15, y + 25),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return frame


# ================= DRAWING FUNCTIONS =================
def draw_point(stroke, point):
    thickness = stroke["thickness"]
    color = stroke["color"]
    stroke_mode = stroke["mode"]

    if stroke_mode == "eraser":
        radius = max(4, thickness)

        cv2.circle(canvas, point, radius, (0, 0, 0), -1)
        cv2.circle(glow_layer, point, radius, (0, 0, 0), -1)

    elif stroke_mode == "neon":
        cv2.circle(
            glow_layer, point,
            max(4, thickness),
            color, -1, cv2.LINE_AA
        )

        cv2.circle(
            canvas, point,
            max(1, thickness // 2),
            (245, 245, 245),
            -1, cv2.LINE_AA
        )

    elif stroke_mode == "rainbow":
        cv2.circle(
            canvas, point,
            max(1, thickness // 2),
            (0, 0, 255),
            -1, cv2.LINE_AA
        )

    else:
        cv2.circle(
            canvas, point,
            max(1, thickness // 2),
            color, -1, cv2.LINE_AA
        )


def draw_segment(stroke, p1, p2):
    thickness = stroke["thickness"]
    color = stroke["color"]
    stroke_mode = stroke["mode"]

    movement = float(np.hypot(
        p2[0] - p1[0],
        p2[1] - p1[1]
    ))

    stroke["distance"] += movement

    if stroke_mode == "eraser":
        radius = max(4, thickness)

        cv2.line(
            canvas, p1, p2,
            (0, 0, 0),
            radius * 2,
            cv2.LINE_AA
        )

        cv2.line(
            glow_layer, p1, p2,
            (0, 0, 0),
            radius * 2,
            cv2.LINE_AA
        )

        cv2.circle(canvas, p2, radius, (0, 0, 0), -1)
        cv2.circle(glow_layer, p2, radius, (0, 0, 0), -1)

    elif stroke_mode == "rainbow":
        hue = (stroke["distance"] * 0.004) % 1.0

        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)

        rainbow_color = (
            int(rgb[2] * 255),
            int(rgb[1] * 255),
            int(rgb[0] * 255)
        )

        cv2.line(
            canvas, p1, p2,
            rainbow_color,
            thickness,
            cv2.LINE_AA
        )

    elif stroke_mode == "neon":
        cv2.line(
            glow_layer, p1, p2,
            color,
            thickness + 10,
            cv2.LINE_AA
        )

        cv2.line(
            canvas, p1, p2,
            color,
            thickness + 2,
            cv2.LINE_AA
        )

        cv2.line(
            canvas, p1, p2,
            (245, 245, 245),
            max(1, thickness // 2),
            cv2.LINE_AA
        )

    else:
        cv2.line(
            canvas, p1, p2,
            color,
            thickness,
            cv2.LINE_AA
        )


# ================= INTERPOLATION =================
def add_interpolated_points(new_point):
    if active_stroke is None:
        return

    previous = active_stroke["points"][-1]

    dx = new_point[0] - previous[0]
    dy = new_point[1] - previous[1]

    distance = float(np.hypot(dx, dy))

    max_step = max(
        2.0,
        active_stroke["thickness"] * 0.5
    )

    steps = max(
        1,
        int(math.ceil(distance / max_step))
    )

    for i in range(1, steps + 1):
        point = (
            int(round(previous[0] + dx * i / steps)),
            int(round(previous[1] + dy * i / steps))
        )

        last_point = active_stroke["points"][-1]

        if point != last_point:
            active_stroke["points"].append(point)

            draw_segment(
                active_stroke,
                last_point,
                point
            )


# ================= REBUILD CANVAS =================
def render_stroke(stroke):
    points = stroke["points"]

    if not points:
        return

    stroke["distance"] = 0.0

    draw_point(stroke, points[0])

    for i in range(1, len(points)):
        draw_segment(
            stroke,
            points[i - 1],
            points[i]
        )


def rebuild_canvas():
    global canvas, glow_layer

    h, w = canvas.shape[:2]

    canvas = np.zeros(
        (h, w, 3),
        dtype=np.uint8
    )

    glow_layer = np.zeros_like(canvas)

    for stroke in strokes:
        render_stroke(stroke)


# ================= FINISH STROKE =================
def finish_stroke():
    global active_stroke, is_drawing

    is_drawing = False

    if active_stroke is not None:
        if active_stroke["points"]:
            strokes.append(active_stroke)

        active_stroke = None


# ================= MAIN LOOP =================
try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        # Original camera image without hand landmarks or UI
        camera_background = frame.copy()

        h, w, _ = frame.shape

        if canvas is None:
            canvas = np.zeros(
                (h, w, 3),
                dtype=np.uint8
            )

            glow_layer = np.zeros_like(canvas)

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]

            thumb_x = int(thumb.x * w)
            thumb_y = int(thumb.y * h)

            index_x = int(index.x * w)
            index_y = int(index.y * h)

            current_point = (index_x, index_y)

            pinch_distance = np.hypot(
                index_x - thumb_x,
                index_y - thumb_y
            )

            # Original smoothing
            if smoothed_point is None:
                smoothed_point = current_point
            else:
                smoothed_point = (
                    int(
                        0.75 * smoothed_point[0]
                        + 0.25 * current_point[0]
                    ),
                    int(
                        0.75 * smoothed_point[1]
                        + 0.25 * current_point[1]
                    )
                )

            # Drawing is completely disabled during pause
            if drawing_paused:
                if is_drawing:
                    finish_stroke()

            else:
                # Start drawing
                if (
                    not is_drawing
                    and pinch_distance < pinch_start_threshold
                ):
                    is_drawing = True
                    redo_strokes.clear()

                    active_stroke = {
                        "points": [smoothed_point],
                        "color": colors[color_index],
                        "thickness": brush_thickness,
                        "mode": mode,
                        "distance": 0.0
                    }

                    draw_point(
                        active_stroke,
                        smoothed_point
                    )

                # Continue drawing
                elif (
                    is_drawing
                    and pinch_distance < pinch_end_threshold
                ):
                    if active_stroke is not None:
                        add_interpolated_points(smoothed_point)

                # Stop drawing
                elif is_drawing:
                    finish_stroke()

            # Cursor
            cursor_color = (
                (200, 200, 200)
                if mode == "eraser"
                else colors[color_index]
            )

            cv2.circle(
                frame,
                smoothed_point,
                7,
                cursor_color,
                -1 if is_drawing else 2,
                cv2.LINE_AA
            )

            if is_drawing:
                cv2.circle(
                    frame,
                    smoothed_point,
                    7,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        else:
            if is_drawing:
                finish_stroke()

            smoothed_point = None

        # ================= GLOW EFFECT =================
        if np.any(glow_layer):
            blurred = cv2.GaussianBlur(
                glow_layer,
                (21, 21),
                0
            )

            display_canvas = cv2.addWeighted(
                canvas,
                1.0,
                blurred,
                0.85,
                0
            )
        else:
            display_canvas = canvas

        # ================= DISPLAY DRAWING =================
        mask = cv2.cvtColor(
            display_canvas,
            cv2.COLOR_BGR2GRAY
        ) > 0

        # Camera + drawing image for F key
        camera_drawing_frame = camera_background.copy()
        camera_drawing_frame[mask] = display_canvas[mask]

        # Normal application display
        frame[mask] = display_canvas[mask]

        frame = draw_ui(frame)

        cv2.imshow("Pinch Drawing", frame)

        # ================= KEYBOARD =================
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # Drawing Pause
        elif key == ord("p"):
            drawing_paused = not drawing_paused

            if drawing_paused:
                finish_stroke()
                show_toast("Drawing paused")
            else:
                show_toast("Drawing resumed")

        elif ord("1") <= key <= ord("9"):
            color_index = key - ord("1")
            mode = "brush"

        elif key == ord("0"):
            color_index = 9
            mode = "brush"

        elif key == ord("b"):
            mode = "brush"

        elif key == ord("e"):
            mode = "eraser"

        elif key == ord("n"):
            mode = "neon"

        elif key == ord("r"):
            mode = "rainbow"

        # Undo
        elif key == ord("u"):
            if active_stroke is not None:
                finish_stroke()

            if strokes:
                redo_strokes.append(strokes.pop())
                rebuild_canvas()

        # Redo
        elif key == ord("y"):
            if active_stroke is not None:
                finish_stroke()

            if redo_strokes:
                strokes.append(redo_strokes.pop())
                rebuild_canvas()

        # Brush size
        elif key in (ord("+"), ord("=")):
            brush_thickness = min(
                brush_thickness + 2,
                50
            )

        elif key == ord("-"):
            brush_thickness = max(
                brush_thickness - 2,
                2
            )

        # Clear confirmation
        elif key == ord("c"):
            now = time.monotonic()

            if now <= clear_armed_until:
                strokes.clear()
                redo_strokes.clear()
                active_stroke = None
                is_drawing = False

                rebuild_canvas()

                clear_armed_until = 0.0
                show_toast("Canvas cleared!")

            else:
                clear_armed_until = now + 2.5
                show_toast("Press C again to clear", 2.5)

        # Save drawing only
        elif key == ord("s"):
            if active_stroke is not None:
                finish_stroke()

            filename = unique_filename("drawing")

            if cv2.imwrite(filename, canvas):
                show_toast("Drawing saved!")
                print(f"Drawing saved as {filename}")
            else:
                show_toast("Save failed!")

        # Save camera + drawing
        elif key == ord("f"):
            if active_stroke is not None:
                finish_stroke()

            filename = unique_filename("camera_drawing")

            if cv2.imwrite(filename, camera_drawing_frame):
                show_toast("Camera image saved!")
                print(f"Camera image saved as {filename}")
            else:
                show_toast("Save failed!")

finally:
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
