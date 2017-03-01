from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from im_persp import warper, warper_Minv
from im_trans import im_trans, undistort
import numpy as np
import cv2

def poly_fit(binary_warped): # from the lecture
    # Create an output image to draw on and  visualize the result
    # out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines

    histogram = np.sum(binary_warped, axis=0)
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    print("left base: ", leftx_base, ", right base: ", rightx_base)
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        # cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        # cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    #
    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.savefig('output_images/lane_poly_fit')
    # plt.show()

    return left_fit, right_fit, leftx, lefty, rightx, righty


def poly_track(binary_warped,left_fit,right_fit):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
    nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
    (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
    nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]


    # Create an image to draw on and an image to show the selection window
    # out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # window_img = np.zeros_like(out_img)
    # # Color in left and right line pixels
    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    # cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    # cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    # result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    # plt.imshow(result)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.savefig('output_images/lane_poly_track.png')
    # plt.show()

    return left_fit, right_fit, leftx, lefty, rightx, righty


def curvature(left_fit, right_fit, leftx, lefty, rightx, righty):
    # curvature
    # leftx = np.array([left_fit[2] + y * left_fit[1] + (y ** 2) * left_fit[0] + np.random.randint(-50, high=51)
    #                   for y in ploty])
    # rightx = np.array([right_fit[2] + y*right_fit[1] + (y ** 2) * right_fit[0] + np.random.randint(-50, high=51)
    #                    for y in ploty])
    # Fit a second order polynomial to each

    # Generate x and y values for plotting
    ploty = np.linspace(0, 719, 720)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Plot up the fake data
    mark_size = 3
    # plt.plot(leftx, lefty, 'o', color='red', markersize=mark_size)
    # plt.plot(rightx, righty, 'o', color='blue', markersize=mark_size)
    # plt.xlim(0, 1280)
    # plt.ylim(0, 720)
    # plt.plot(left_fitx, ploty, color='green', linewidth=3)
    # plt.plot(right_fitx, ploty, color='green', linewidth=3)
    # plt.gca().invert_yaxis()  # to visualize as we do the images
    # plt.savefig('output_images/lane_curvature.png')
    # plt.show()

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
    print("curvature in pixel", left_curverad, right_curverad)

    offset = (left_fitx[-1] + right_fitx[-1])/2-1280/2 # + right - left
    print("center bias in pixel", offset)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / np.mean(right_fitx-left_fitx)  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    print("curvature in meter", left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m

    offset_dist = offset*xm_per_pix
    print("offset in meter ", offset_dist, 'm')

    return left_curverad, right_curverad, offset_dist


def lane_plot(undist, warped, left_fit,right_fit):
    # Create an image to draw the lines on
    print("size")
    print(warped.shape)
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    Minv = warper_Minv(warped)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    return result

def lane_fit(img):
    undist = undistort(img)
    binary = im_trans(img)
    # warped = warper(img)
    binary_warped = warper(binary)
    left_fit, right_fit, leftx, lefty, rightx, righty = poly_fit(binary_warped)
    left_fit, right_fit, leftx, lefty, rightx, righty = poly_track(binary_warped, left_fit, right_fit)
    left_curverad, right_curverad, offset_dist = curvature(left_fit, right_fit, leftx, lefty, rightx, righty)
    result = lane_plot(undist, binary_warped, left_fit, right_fit)
    return result

if __name__ == '__main__':
    # ori_img = mpimg.imread('test_images/straight_lines2.jpg')
    image = mpimg.imread('test_images/test2.jpg')
    undist = undistort(image)
    binary = im_trans(image)
    warped = warper(image)
    binary_warped = warper(binary)

    # f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(24,9))
    # f.tight_layout()
    # ax1.imshow(image,cmap='gray')
    # ax1.set_title("Original image", fontsize=40)
    # ax2.imshow(binary,cmap='gray')
    # ax2.set_title("Binary image", fontsize=40)
    # ax3.imshow(binary_warped,cmap='gray')
    # ax3.set_title("Warped image", fontsize=40)
    # plt.savefig('output_images/lane_binary_warped.png')
    # plt.show()

    histogram = np.sum(binary_warped,axis=0)
    #histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)

    # f, (ax1,ax2) = plt.subplots(1,2,figsize=(24,9))
    # f.tight_layout()
    # ax1.imshow(binary_warped,cmap='gray')
    # ax1.set_title("Warped image", fontsize=40)
    # ax2.plot(histogram)
    # ax2.set_title("Histogram", fontsize=40)
    # plt.savefig('output_images/lane_histogram.png')
    # plt.show()

    # fit poly
    left_fit, right_fit, leftx, lefty, rightx, righty = poly_fit(binary_warped)
    left_fit, right_fit, leftx, lefty, rightx, righty = poly_track(binary_warped,left_fit,right_fit)
    left_curverad, right_curverad, offset_dist = curvature(left_fit, right_fit, leftx, lefty, rightx, righty)
    result = lane_plot(undist, binary_warped,left_fit,right_fit)

    f, ax = plt.subplots()
    f.tight_layout()
    ax.imshow(result)
    ax.text(50,30,"Radius of Curvature = {:4.0f} m".format(left_curverad),ha='left',va='top',color='white',fontsize=20)
    ax.text(50,100, "Vehicle is {:.02f} m right of center".format(offset_dist), ha='left', va='top', color='white',
            fontsize=20)
    plt.savefig('output_images/lane_final.png')
    plt.show()