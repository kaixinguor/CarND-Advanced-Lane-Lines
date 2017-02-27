import math
import cv2
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
import numpy as np


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
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

    cv2.imshow("test",mask)
    cv2.waitKey()

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image, mask


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    #    for line in lines:
    #        for x1,y1,x2,y2 in line:
    #            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    n_left = 0
    k_left = 0
    b_left = 0
    n_right = 0
    k_right = 0
    b_right = 0
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y1 - y2) / (x1 - x2)
            b = 0.5 * (y1 + y2 - k * (x1 + x2))
            if k < 0:
                k_left = k_left + k
                b_left = b_left + b
                n_left = n_left + 1
            else:
                k_right = k_right + k
                b_right = b_right + b
                n_right = n_right + 1
    k_left = k_left / n_left
    b_left = b_left / n_left
    k_right = k_right / n_right
    b_right = b_right / n_right

    y1 = int(round(img.shape[0] * 0.6))
    y2 = img.shape[0]
    x1 = int(round((y1 - b_left) / k_left))
    x2 = int(round((y2 - b_left) / k_left))
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    x1 = int(round((y1 - b_right) / k_right))
    x2 = int(round((y2 - b_right) / k_right))
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, [255, 0, 0], thickness=10)
    # print("{0:d} lines detected".format(len(lines)))
    # print(lines)

    # transparent?
    # weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def process_image1(image):
    # convert to gray scale image
    gray = grayscale(image)

    # Gaussian smoothing / blurring
    kernel_size = 3
    blur_gray = gaussian_blur(gray, kernel_size)

    # Canny edge detection
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    # cv2.imshow("test",edges)
    # cv2.waitKey()

    # region of interest
    imshape = image.shape
    vertices = np.array([[(imshape[1] * 0.12, imshape[0]), (imshape[1] * 0.45, imshape[0] * 0.6), \
                          (imshape[1] * 0.55, imshape[0] * 0.6), (imshape[1] * 0.95, imshape[0])]], dtype=np.int32)

    # draw roi
    num_pt = len(vertices[0, :])
    roi_image = np.copy(image)
    for i in np.arange(num_pt):
        pt1 = vertices[0, i]
        pt2 = vertices[0, i - 3]
        cv2.line(roi_image, (pt1[0], pt1[1]), (pt2[0], pt2[1]), [0, 0, 255], 5)

    masked_edges = region_of_interest(edges, vertices)

    # cv2.imshow("test",masked_edges)
    # cv2.waitKey()

    # Hough transform
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = 1 * np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 40  # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 20  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on
    lines = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges))

    # Draw the lines on the edge image
    lines_edges = weighted_img(color_edges, lines)
    result_image = weighted_img(image, lines)
    return result_image, lines_edges, roi_image

if __name__ == "__main__":

    image = mpimg.imread('test_images/straight_lines2.jpg')
    # image = mpimg.imread('test_images/test1.jpg')

    result_image, lines_edges, roi_image = process_image1(image)
    print(result_image)
    # plot
    f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(24,9))
    f.tight_layout()
    ax1.imshow(roi_image,cmap='gray')
    ax1.set_title("undistorted")
    ax2.imshow(lines_edges,cmap='gray')
    ax2.set_title("sobel threshold")
    ax3.imshow(result_image,cmap='gray')
    ax3.set_title("color threshold")
    plt.show()