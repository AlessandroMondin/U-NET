import os.path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

parent_folder = "/path/to/your/folder/of/chessboard/images"
# select all the jpg or png images
images = glob.glob(os.path.join(parent_folder, '*.jpg'))

# creates object points and image points
# glob_image_folder is the folder with your chessboard image
# nx is the number of squares in the vertical side of the chessboard
# nx is the number of squares in the horizontal side of the chessboard

def calibrate_camera(glob_image_folder, ny, nx):
    objpoints = []
    imgpoints = []
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:ny, 0:nx].T.reshape(-1, 2)
    for image in glob_image_folder:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (ny, nx), None)
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    return mtx, dist

# mtx and dist are geometrical retrieved by the function here above
def plot_calibration(glob_image_folder, mtx, dist):
    # can be loaded only 20 images per time due to RAM
    if len(glob_image_folder) > 20:
        i = 20
    else:
        i = len(glob_image_folder)
    for image in glob_image_folder[:i]:
        img = cv2.imread(image)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        undist_img = cv2.undistort(img, mtx, dist, None, mtx)
        ax1.set_title('Original Image', fontsize=15)
        ax1.imshow(img)
        ax2.set_title('Undistorted Image', fontsize=15)
        ax2.imshow(undist_img)
    plt.show()

mtx, dist = calibrate_camera(images, 8, 6)
plot_calibration(images, mtx, dist)
