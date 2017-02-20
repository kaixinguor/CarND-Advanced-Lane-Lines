import cv2
import numpy as np
import os
from myutil import ensure_dir
from matplotlib import pyplot as plt

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


def gradient_thresh(img, orient='x', thresh=(0,255)):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if orient == 'x' or orient == 'y':
        if orient == 'x':
            sobel_img = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        elif orient == 'y':
            sobel_img = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        abs_sobel = np.absolute(sobel_img)
    elif orient == 'xy':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        abs_sobel = np.sqrt(sobelx**2+sobely**2)
    else:
        print("error")
        return

    scale_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    sbinary = np.zeros_like(gray)
    sbinary[(scale_sobel >= thresh[0]) & (scale_sobel <= thresh[1])] = 1

    return sbinary, abs_sobel, scale_sobel


def hls_thresh(img, channel='s', thresh=(0,255)):

    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    s_img = hls_img[:, :, 'hls'.find(channel)]
    s_binary = np.zeros_like(s_img)
    s_binary[(s_img>=thresh[0]) & (s_img<=thresh[1])] = 1
    return s_binary, s_img

if __name__ == '__main__':

    image_file = 'test_images/test2.jpg'
    ori_img = cv2.imread(image_file)
    img = cv2.cvtColor(ori_img,cv2.COLOR_BGR2RGB) # convert to RGB

    plt.subplot(1,3,1)
    plt.imshow(img)


    # compute gradient
    sbinary, abs_sobel, scale_sobel = gradient_thresh(img,orient='x',thresh=(20,100))
    print(np.max(abs_sobel))

    plt.subplot(1,3,2)
    plt.imshow(sbinary,cmap='gray')


    # color transform
    hls_binary, s_img = hls_thresh(img, channel='s',thresh=(90,255))
    print(s_img.shape)
    print(hls_binary.shape)

    plt.subplot(1, 3, 3)
    plt.imshow(hls_binary,cmap='gray')

    plt.show()

