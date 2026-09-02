"""
main.py
-------
AirSketch3D entry point.

Hand landmark reference (MediaPipe):
    0  = wrist
    4  = thumb tip
    8  = index fingertip
    12 = middle fingertip
    16 = ring fingertip
    20 = pinky fingertip

Controls:
    Index finger only extended   -> draw on the canvas
    Index + middle extended      -> selection mode (pick color/eraser, no drawing)
    '3' key                      -> convert current 2D drawing to a 3D mesh and preview it
    'c' key                      -> clear the canvas
    's' key                      -> save the current 2D drawing as a PNG
    'q' key                      -> quit
"""

import cv2
import time

from hand_tracker import HandTracker
from canvas import Canvas
from convert_3d import convert_points_to_mesh, export_mesh


def run():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not open webcam.")
        return
    height, width, _ = frame.shape

    tracker = HandTracker(max_hands=1)
    canvas = Canvas(width, height)

    mode = "2D"  # "2D" or "3D"
    prev_time = 0

    print("AirSketch3D running. Press '3' to convert to 3D, 'c' to clear, 's' to save, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame so movement feels natural (like a mirror, not a camera)
        frame = cv2.flip(frame, 1)

        if mode == "2D":
            frame = tracker.find_hands(frame, draw=False)
            landmarks = tracker.get_landmark_positions(frame)
            gesture, fingers = tracker.get_gesture(landmarks)

            if landmarks:
                # Landmark 8 = index fingertip; this is our drawing/pointer position
                _, x8, y8 = landmarks[8]

                if gesture == "select":
                    # Selection mode: check if hovering over a palette/eraser button
                    canvas.check_toolbar_hover(x8, y8)
                    canvas.end_stroke()  # don't connect a line across mode switches
                    cv2.circle(frame, (x8, y8), 10, (255, 255, 255), cv2.FILLED)

                elif gesture == "draw":
                    canvas.start_or_continue_stroke(x8, y8)
                    cv2.circle(frame, (x8, y8), 8, canvas.draw_color, cv2.FILLED)

                else:
                    # Any other pose: stop the current stroke so lines don't jump
                    canvas.end_stroke()
            else:
                canvas.end_stroke()

            frame = canvas.composite(frame)
            frame = canvas.draw_toolbar(frame)

            # FPS counter (helpful for debugging performance)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (width - 150, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "Mode: 2D Draw", (20, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("AirSketch3D", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('c'):
            canvas.clear()
            print("Canvas cleared.")

        elif key == ord('s'):
            path = canvas.save("drawing.png")
            print(f"Drawing saved to {path}")

        elif key == ord('3'):
            canvas.end_stroke()  # make sure the in-progress stroke is saved
            points = canvas.get_all_points()
            if len(points) < 2:
                print("Draw a shape with at least 2 points before converting to 3D.")
                continue

            print(f"Converting 2D drawing to 3D mesh ({len(points)} points)...")
            try:
                mesh = convert_points_to_mesh(
                    points, canvas_width=width, canvas_height=height, extrude_height=50.0
                )
                out_path = export_mesh(mesh, "airsketch_model.obj")
                print(f"3D mesh exported to {out_path}")

                # Open the simple desktop 3D viewer so the user can rotate/inspect it.
                # This call blocks until the viewer window is closed.
                mesh.show()

            except Exception as e:
                # Catch EVERYTHING here (not just ValueError) so a viewer/
                # dependency problem prints a message instead of silently
                # killing the whole app.
                import traceback
                print("Could not build/show 3D mesh:")
                traceback.print_exc()

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    run()
