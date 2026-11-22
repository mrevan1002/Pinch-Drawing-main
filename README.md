# Pinch-Drawing
Hand gesture drawing application using Python and OpenCV

## About the Project

Pinch Drawing allows users to draw on their computer screen using hand gestures through a webcam.

By bringing the thumb and index finger together, users can draw freely in the air.

## Features

- Real-time hand tracking
- Finger pinch gesture recognition
- Smooth drawing with coordinate smoothing and interpolation
- Multiple drawing modes: Brush, Eraser, Neon and Rainbow
- Multiple color options
- Adjustable brush thickness
- Undo and Redo functionality
- Pause and Resume functionality
- Clear canvas functionality
- Save drawing with timestamped filenames
- Save drawing with or without the webcam background
- On-screen notifications for actions

## Technologies Used

- Python 3.12
- OpenCV
- MediaPipe
- NumPy

## How to Run

1. Install Python 3.12.
2. Install the required libraries:

```bash
pip install opencv-python mediapipe==0.10.21 numpy
```

3. Run the application:

```bash
py -3.12 pinch_drawing.py
```

## Controls

| Key | Action |
|---|---|
| Pinch fingers together | Draw |
| `+` / `-` | Increase / Decrease brush thickness |
| `U` | Undo |
| `Y` | Redo |
| `C` | Clear canvas (press twice to confirm) |
| `P` | Pause / Resume |
| `S` | Save drawing only |
| `F` | Save drawing with webcam background |
| `Q` | Quit application |

## Saving Drawings

- Press `S` to save the drawing on its own.
- Press `F` to save the drawing with the webcam background.
- Saved images use timestamped filenames to help prevent overwriting previous drawings.

## Project Structure

```text
Pinch-Drawing/
├── pinch_drawing.py
├── README.md
└── .gitignore
```

---

Created as a Python computer vision project.