import numpy as np
import cv2
import glob
import os

def ensure_dir(d):
    if not os.path.exists(d):
        os.mkdirs(d)

def find_corners(img,pattern):
    # adapted from http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html


    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, pattern, None)

    # If found, add object points, image points (after refining them)

    if ret == True:

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, pattern, corners,ret)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((pattern[0]*pattern[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern[0],0:pattern[1]].T.reshape(-1,2)

    return ret, corners, objp, pattern


if __name__ == '__main__':

    patterns = [(9,6), (9,5), (7,6), (6,6)] # [1,17,1,1]

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('camera_cal/*.jpg')
    out_path = 'output_images/calib'
    ensure_dir(out_path)

    for fname in images:
    #fname = 'camera_cal/calibration4.jpg'
        img = cv2.imread(fname)

        for pattern in patterns:
            ret, corners, objp, found_pattern = find_corners(img, pattern)
            if ret == True:
                break

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, found_pattern, corners,ret)
            cv2.imwrite(os.path.join(out_path,fname[11:]),img)

        else:
            print fname
            print ret

    for objpoint in objpoints:
        print objpoint.shape

    for imgpoint in imgpoints:
        print imgpoint.shape
    cv2.destroyAllWindows()