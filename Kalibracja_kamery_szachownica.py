import numpy as np
import cv2
import glob
from pathlib import Path

GRID_SIZE = (15, 10)


# GRID_SIZE = (23, 16)

def main():
    CALIBRATION_DIR = "./frames/"
    dir = Path(CALIBRATION_DIR)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((GRID_SIZE[0] * GRID_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:GRID_SIZE[0], 0:GRID_SIZE[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []
    image_shape = None

    images = glob.glob(f"{str(dir)}/*.png")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, GRID_SIZE, None)

        if ret:
            if image_shape is None:
                image_shape = gray.shape

            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            cv2.drawChessboardCorners(img, GRID_SIZE, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(200)

    cv2.destroyAllWindows()

    if not objpoints:
        print("Nie znaleziono żadnych szachownic. Kalibracja przerwana.")
        return

    print(f'Kalibracja')

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_shape[::-1], None, None
    )

    h, w = image_shape
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    print(f'Kalibracja zakończona.')
    print(f'Camera matrix:\n{mtx}\n')
    print(f'Distortion coefficients:\n{dist}\n')
    print(f'New camera matrix:\n{newcameramtx}\n')
    print(f'ROI:\n{roi}\n')


if __name__ == "__main__":
    main()
