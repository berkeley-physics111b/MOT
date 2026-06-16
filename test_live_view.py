"""
test_live_view.py
==================
Minimal test for allied_vision_camera.AlliedVisionCamera.

Opens the camera, grabs a single live frame, displays it in an OpenCV
window, and closes the camera.

Run:
    python test_live_view.py
Press any key (with the image window focused) to close.
"""

import sys

import cv2

from allied_vision_camera import AlliedVisionCamera, CameraConfig


def main() -> int:
    with AlliedVisionCamera(CameraConfig()) as cam:
        print("Camera info:", cam.get_camera_info())

        frame_holder = {"img": None}

        def _on_frame(img, ts):
            frame_holder["img"] = img

        print("Starting stream, waiting for first frame...")
        cam.start_continuous(callback=_on_frame)

        # Wait briefly for the first frame to arrive
        for _ in range(50):  # up to ~5 s
            if frame_holder["img"] is not None:
                break
            cv2.waitKey(100)
        else:
            print("ERROR: no frame received within timeout.")
            cam.stop_continuous()
            return 1

        cam.stop_continuous()

        print(f"Got frame: shape={frame_holder['img'].shape}")
        cv2.imshow("Live Image", frame_holder["img"])
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())