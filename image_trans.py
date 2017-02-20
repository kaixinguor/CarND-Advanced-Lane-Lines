import cv2
import numpy as np
import os
from myutil import ensure_dir


def undistort():
    path = 'output_images'
    calib = np.load(os.path.join(path, 'calib_mtx.npz'))

    mtx = calib['mtx']
    dist = calib['dist']

    test_image = 'test_images/test1.jpg'
    img = cv2.imread(test_image)
    undistort_img = cv2.undistort(img, mtx, dist)

    # undistort
    out_path = 'output_images/undistort_lane'
    ensure_dir(out_path)
    cv2.imwrite(os.path.join(out_path, 'test1.jpg'), undistort_img)


if __name__ == '__main__':


    test_image = 'test_images/test2.jpg'
    img = cv2.imread(test_image)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow("test",gray)
    cv2.waitKey()

    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0)
    abs_sobelx = np.absolute(sobelx)
    scale_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0,1)
    abs_sobely = np.absolute(sobely)
    scale_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))

    cv2.imshow("test",scale_sobelx)
    cv2.waitKey()
    cv2.imshow("test",scale_sobely)
    cv2.waitKey()
    print(np.max(abs_sobelx))

    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(gray)
    sxbinary[(scale_sobelx>=thresh_min) & (scale_sobelx<=thresh_max)] = 1
    cv2.imshow("test", sxbinary)
    cv2.waitKey()
    # compute gradient


    # color transform
    hsl_img = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    s_img = hsl_img[:,:,2]
    cv2.imshow("test",s_img)
    cv2.waitKey()

