import cv2
import numpy as np
import os
from myutil import ensure_dir
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

def undistort(img):
    path = 'output_images'
    calib = np.load(os.path.join(path, 'calib_mtx.npz'))

    mtx = calib['mtx']
    dist = calib['dist']

    undistort_img = cv2.undistort(img, mtx, dist)

    return undistort_img


def gradient_thresh(img, orient='x', thresh=(0,255)):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if orient == 'x' or orient == 'y':
        if orient == 'x':
            sobel_img = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        else:
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

    # f, (ax1,ax2) = plt.subplots(1,2,figsize=(24,9))
    # f.tight_layout()
    # ax1.imshow(scale_sobel,cmap='gray')
    # ax1.set_title("scale sobel ")
    # ax2.imshow(sbinary,cmap='gray')
    # ax2.set_title("sobel binary")
    # plt.show()

    return sbinary, abs_sobel, scale_sobel


def hls_thresh(img, channel='s', thresh=(0,255)):

    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    # plt.subplot(1,3,1)
    # plt.imshow(hls_img[:,:,0],cmap='gray')
    # plt.subplot(1,3,2)
    # plt.imshow(hls_img[:,:,1],cmap='gray')
    # plt.subplot(1,3,3)
    # plt.imshow(hls_img[:,:,2],cmap='gray')
    # plt.show()

    s_img = hls_img[:, :, 'hls'.find(channel)]
    s_binary = np.zeros_like(s_img)
    s_binary[(s_img>=thresh[0]) & (s_img<=thresh[1])] = 1

    # plt.subplot(1,2,1)
    # plt.imshow(s_img,cmap='gray')
    # plt.title("s channel")
    # plt.subplot(1,2,2)
    # plt.imshow(s_binary,cmap='gray')
    # plt.title("s binary")
    # plt.show()

    return s_binary, s_img

def roi(img):
    """
    Defines an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """

    imshape = img.shape
    # vertices = np.array([[(imshape[1] * 0.12, imshape[0]), (imshape[1] * 0.45, imshape[0] * 0.6),
    #                       (imshape[1] * 0.55, imshape[0] * 0.6), (imshape[1] * 0.95, imshape[0])]], dtype=np.int32)

    vertices = np.array([[(imshape[1] * 0.15+10, imshape[0]), (imshape[1] * 0.46+20, imshape[0] * 0.6),
                          (imshape[1] * 0.54-20, imshape[0] * 0.6), (imshape[1] * 0.85+40, imshape[0])]], dtype=np.int32)
    # draw roi
    num_pt = len(vertices[0, :])
    roi_plt = np.copy(img)
    for i in np.arange(num_pt):
        pt1 = vertices[0, i]
        pt2 = vertices[0, i - 3]
        cv2.line(roi_plt, (pt1[0], pt1[1]), (pt2[0], pt2[1]), [0, 0, 255], 5)

    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return mask,masked_image,roi_plt


def warper(img):
    pass

if __name__ == '__main__':
    ori_img = mpimg.imread('test_images/straight_lines2.jpg')
    # ori_img = mpimg.imread('test_images/test1.jpg')

    # undistort
    undistort_img = undistort(ori_img)
    #bgr_img = cv2.cvtColor(undistort_img, cv2.COLOR_RGB2BGR)
    plt.imsave('output_images/test_undistort.jpg', undistort_img)

    # compute gradient
    sbinary, abs_sobel, scale_sobel = gradient_thresh(undistort_img,orient='x',thresh=(10, 100))
    print(np.max(abs_sobel))

    # color transform
    hls_binary, s_img = hls_thresh(undistort_img, channel='s',thresh=(90,255))
    print(s_img.shape)
    print(hls_binary.shape)

    # plot
    # f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(24,9))
    # f.tight_layout()
    # ax1.imshow(undistort_img,cmap='gray')
    # ax1.set_title("undistorted")
    # ax2.imshow(sbinary,cmap='gray')
    # ax2.set_title("sobel threshold")
    # ax3.imshow(hls_binary,cmap='gray')
    # ax3.set_title("color threshold")
    # plt.show()
    #plt.savefig('output_images/test_binary.jpg')??

    # combine two binary images
    # binary = sbinary & hls_binary
    # plt.imshow(binary,cmap='gray')
    # plt.show()

    # region of interest
    roi_mask, roi_masked_img, roi_plt = roi(ori_img)
    f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(24,9))
    f.tight_layout()
    ax1.imshow(undistort_img,cmap='gray')
    ax1.set_title("undistorted")
    ax2.imshow(roi_masked_img,cmap='gray')
    ax2.set_title("roi image")
    ax3.imshow(roi_plt)
    ax3.set_title("roi region")
    plt.show()

    # binary = roi_mask & hls_binary
    # plt.imshow(binary,cmap='gray')
    # plt.show()