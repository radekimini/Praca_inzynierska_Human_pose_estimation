import numpy as np
import cv2

# w sali w poniedzialek 30.06 mala dzwi od szafy

# cfg = {
#     'camera_matrix': [
#         [1.48981430e+03, 0.00000000e+00, 9.47488444e+02],
#         [0.00000000e+00, 1.49292142e+03, 5.11038689e+02],
#     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
#     ],
#     'dist_matrix': [
# [ 0.12908319, -0.91693538,  0.00140281,  0.00240981,  1.33891196]
#     ],
#     'new_camera_matrix': [[1.47782284e+03, 0.00000000e+00, 9.52451205e+02],
#  [0.00000000e+00, 1.47498039e+03, 5.12338275e+02],
#  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
#     'roi': [21, 12, 1875, 1049]
# }

#31.07 podczas oddawania

cfg = {
    'camera_matrix': [
        [659.1335331,    0.,         309.26599601],
        [0.,         658.13176293, 225.12565591],
    [0.,           0.,           1.]
    ],
    'dist_matrix': [
[ 1.20261687e-01, -7.89795463e-01,  2.13838781e-03, -2.47974967e-05, 1.03624635e+00]
    ],
    'new_camera_matrix': [[651.0860892,    0.,         309.02327403],
 [  0.,         650.56864018, 225.98281995],
 [  0.,           0.,           1.        ]],
    'roi': [3, 4, 632, 471]
}


def main():
    input_path = "take14_Radek_Kibic/output.mp4"  # 🔄 <- zmień na swoją ścieżkę
    output_path = "take14_Radek_Kibic/Take14_Calibrated.mp4"

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("❌ Nie można otworzyć pliku wideo.")
        return

    # Parametry wideo
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Możesz też dopasować rozmiar do przyciętej wersji
    x, y, w, h = cfg['roi']
    out_size = (w, h)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Można użyć też 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, out_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Korekcja dystorsji
        frame = cv2.undistort(frame, np.array(cfg['camera_matrix']), np.array(cfg['dist_matrix']), None,
                              np.array(cfg['new_camera_matrix']))

        # Przycięcie obrazu do ROI
        frame = frame[y:y + h, x:x + w]

        out.write(frame)

    cap.release()
    out.release()
    print("✅ Zapisano przetworzony film jako:", output_path)

if __name__ == "__main__":
    main()
