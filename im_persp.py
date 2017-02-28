import cv2
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
import numpy as np

def get_corr_pts(img):

    # region of interest
    num_pt = 4

    imsize = (img.shape[1],img.shape[0])

    src_pts = np.array([[(imsize[0]/6+10, imsize[1]),
                         (imsize[0] / 2 - 60, imsize[1] / 2 + 100),
                         (imsize[0] / 2 + 65, imsize[1] / 2 + 100),
                         (imsize[0]*5/6 + 40, imsize[1])]],dtype=np.float32)

    dst_pts = np.array([[(imsize[0]/4,imsize[1]),(imsize[0]/4, 0),
                         (imsize[0]*3/4,0),(imsize[0]*3/4,imsize[1])]],dtype=np.float32)

    return num_pt, src_pts, dst_pts

def warper(img):
    print(img.shape)
    num_pt, src_pts, dst_pts = get_corr_pts(img)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)
    return warped

if __name__ == "__main__":
    #image = mpimg.imread('test_images/straight_lines2.jpg')
    image = mpimg.imread('test_images/test2.jpg')

    num_pt, src_pts, dst_pts = get_corr_pts(image)
    print(src_pts)
    print(dst_pts)

    warped = warper(image)

    # fix source points
    img_copy = image.copy()
    for i in np.arange(num_pt):
        pt1 = src_pts[0, i]
        pt2 = src_pts[0, i - 3]
        cv2.line(img_copy, (pt1[0], pt1[1]), (pt2[0], pt2[1]), [0, 0, 255], 5)

    img_copy2 = warped.copy()
    for i in np.arange(num_pt):
        pt1 = dst_pts[0, i]
        pt2 = dst_pts[0, i - 3]
        cv2.line(img_copy2, (pt1[0], pt1[1]), (pt2[0], pt2[1]), [0, 0, 255], 5)

    f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(24,9))
    f.tight_layout()
    ax1.imshow(image,cmap='gray')
    ax1.set_title("original image", fontsize=40)
    ax2.imshow(img_copy,cmap='gray')
    ax2.set_title("src_pts drawn", fontsize=40)
    ax3.imshow(img_copy2)
    ax3.set_title("warped image with dst_pts", fontsize=40)
    plt.savefig('output_images/test_persp.png')
    plt.show()