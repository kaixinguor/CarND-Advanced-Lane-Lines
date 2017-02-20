import numpy as np
import cv2
import glob
import os
from myutil import ensure_dir


def find_corners(img,pattern):
    # adapted from http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, pattern, None)
    if ret is True:
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((pattern[0]*pattern[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern[0],0:pattern[1]].T.reshape(-1,2)

    return ret, corners, objp, pattern


def find_pts(out_path='output_images/calib'):
    ensure_dir(out_path)

    # calib patterns
    patterns = [(9,6), (9,5), (7,6), (4,6)] # [X1(1.jpg),X17,X1(5.jpg),X1(4.jpg)] 6X6 if change direction for 4.jpg

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('camera_cal/*.jpg')

    for fname in images:
        # fname = 'camera_cal/calibration4.jpg'
        img = cv2.imread(fname)
        print(fname)
        print(img.shape)

        for pattern in patterns:
            ret, corners, objp, found_pattern = find_corners(img, pattern)
            if ret == True:
                break

        # If found, add object points, image points (after refining them)
        if ret is True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, found_pattern, corners,ret)
            cv2.imwrite(os.path.join(out_path,fname[11:]),img)

        else:
            print("corner points detection failure")
            print(fname)
            print(ret)
    imgsize = img.shape
    print(imgsize)

    return objpoints, imgpoints, imgsize

def cam_calib(objpoints,imgpoints,imgsize):

    # calibration
    cret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints,(imgsize[1],imgsize[0]),None,None)

    return mtx,dist

if __name__ == '__main__':

    path = 'output_images'
    ensure_dir(path)

    # find corners
    objpoints, imgpoints, imgsize = find_pts()
    np.savez(os.path.join(path, 'calib_pts.npz'),objpoints=objpoints,imgpoints=imgpoints,imgsize=imgsize)

    # calibration
    mtx,dist = cam_calib(objpoints,imgpoints,imgsize)
    np.savez(os.path.join(path, 'calib_mtx.npz'), mtx=mtx, dist=dist)

    # undistort
    out_path = 'output_images/undistort'
    ensure_dir(out_path)

    images = glob.glob('camera_cal/*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        undistort_img = cv2.undistort(img, mtx, dist)
        cv2.imwrite(os.path.join(out_path, 'undistort'+fname[22:]), undistort_img)