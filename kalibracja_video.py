import numpy as np
import cv2

cfg = {
    'camera_matrix': [
        [659.1335331, 0., 309.26599601],
        [0., 658.13176293, 225.12565591],
        [0., 0., 1.]
    ],
    'dist_matrix': [
        [1.20261687e-01, -7.89795463e-01, 2.13838781e-03, -2.47974967e-05, 1.03624635e+00]
    ],
    'new_camera_matrix': [[651.0860892, 0., 309.02327403],
                          [0., 650.56864018, 225.98281995],
                          [0., 0., 1.]],
    'roi': [3, 4, 632, 471]
}


def main():
    input_path = "ścieszka z plikami"
    output_path = "ścieszka do zapisu"

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Nie można otworzyć pliku wideo.")
        return

    # Parametry wideo
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    x, y, w, h = cfg['roi']
    out_size = (w, h)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, out_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.undistort(frame, np.array(cfg['camera_matrix']), np.array(cfg['dist_matrix']), None,
                              np.array(cfg['new_camera_matrix']))

        frame = frame[y:y + h, x:x + w]

        out.write(frame)

    cap.release()
    out.release()
    print("Zapisano film jako:", output_path)


if __name__ == "__main__":
    main()
