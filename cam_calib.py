import numpy as np
import cv2
import glob
from PIL import Image
from matplotlib import pyplot as plt
import os
from myutil import ensure_dir

"""
    code for doing camera calibration
    reference: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html
"""


def find_corners(img,pattern):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, pattern, None)
    if ret is True:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # termination criteria
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((pattern[0]*pattern[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern[0],0:pattern[1]].T.reshape(-1,2)

    return ret, corners, objp, pattern


def find_pts(imgs,patterns):

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    sizes = []

    nImg = len(imgs)
    corner_imgs = []
    sel_patterns = np.zeros(nImg)
    for i in range(nImg):
        img = imgs[i].copy()
        sizes.append(img.shape)

        for j in range(len(patterns)):
            pattern = patterns[j]
            ret, corners, objp, found_pattern = find_corners(img, pattern)
            if ret is True:
                sel_patterns[i] = j
                break

        # If found, add object points, image points (after refining them)
        if ret is True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, found_pattern, corners,ret)
            corner_imgs.append(img)
            print("succeed")
            print(i)
        else:
            print("corner points detection failure")
            print(i)

    return objpoints, imgpoints, sizes, corner_imgs, sel_patterns


def cam_calib(objpoints,imgpoints,imgsize):

    # calibration


    return mtx,dist

if __name__ == '__main__':

    imagefiles = glob.glob('camera_cal/*.jpg') # unsorted list
    print(imagefiles[14])
    images = []
    for f in imagefiles:
        image = cv2.imread(f)
        images.append(image)

    # define calib corner patterns
    patterns = [(9,6), (9,5), (7,6), (6,5)] # [X1(1.jpg),X17,X1(5.jpg),X1(4.jpg)] 6X6 if change direction for 4.jpg

    # find corners
    objpoints, imgpoints, sizes, corner_imgs, sel_patterns = find_pts(images, patterns)
    print(imagefiles)
    print(sizes)
    print(sel_patterns)

    imsize = (sizes[0][1],sizes[0][0])
    print(imsize)

    # calibration
    cret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imsize, None, None)

    # undistort
    undistort_imgs = []
    for image in images:
        undistort_img = cv2.undistort(image, mtx, dist)
        undistort_imgs.append(undistort_img)

    # save result
    np.savez('output_images/calib_pts.npz', objpoints=objpoints, imgpoints=imgpoints, imgsize=imsize)
    np.savez('output_images/calib_mtx.npz', mtx=mtx, dist=dist)

    # plot
    f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(24,9))
    f.tight_layout()
    ax1.imshow(images[0])
    ax1.set_title("Original image",fontsize=40)
    ax2.imshow(corner_imgs[0])
    ax2.set_title("Found corners",fontsize=40)
    ax3.imshow(undistort_imgs[0])
    ax3.set_title("Undistorted image",fontsize=40)
    plt.savefig('output_images/cam_calib.png')
    plt.show()