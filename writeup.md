##Advanced Lane Finding Project

###The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image_1_0]: ./output_images/cam_calib.png "Camera calibration"
[image_2_0]: ./test_images/test2.jpg "Road Transformed"
[image_2_1]: ./output_images/test_undistort.png "Undistort"
[image_2_2_1]: ./output_images/test_sobel.png "gradient"
[image_2_2_2]: ./output_images/test_sobel_binary.png "gradient"
[image_2_3_1]: ./output_images/test_hls.png "color"
[image_2_3_2]: ./output_images/test_s_binary.png "color"
[image_2_4_1]: ./output_images/test_combine.png "combine"
[image_2_4_2]: ./output_images/test_final.png "final"
[image_2_5_1]: ./output_images/test_persp_straight2.png "straght lane"
[image_2_5_2]: ./output_images/test_persp_test2.png "curved lane"
[image_2_6_1]: ./output_images/lane_binary_warped.png "Warped"
[image_2_6_2]: ./output_images/lane_poly_fit.png "Fit"
[image_2_6_3]: ./output_images/lane_poly_track.png "Track"
[image_2_6_4]: ./output_images/lane_curvature.png "Curvature"
[image_2_7]: ./output_images/lane_final.png "Final"

####Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---


###Camera Calibration

####1. Computed camera matrix and distortion coefficients from given chessboard images. 


The code is in the file `./cam_calib.py`; the calibration data is in the files `/output_images/calib_pts.npz` and `/output_images/calib_mtx.npz`  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

For detecting chessboard corners, I first convert each chessboard image into gray scale, then use `cv2.findChessboardCorners()` to detect corner points. The points are plotted in the image by `cv2.drawChessboardCorners()`. 

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the `Original image` using the `cv2.undistort()` function.

The following shows the `Original image`, the `Found corner` image and the `Undistorted image`.

![alt text][image_1_0]

I have found two interesting issues during finding corners: 

* The usually pattern is set to (9,6), however some times opencv fails to detect given pattern. 

An example is the above image, it gets 9X5 corners, however, opencv will simply return a false when the pattern is set to (9,6). This may due to that opencv is not very flexible for input patterns.

* When rotating the image by 90 degree (image4), the patterns which could be detected are different.
 
---

###Pipeline (single images)

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image_2_0]

####1. Provide an example of a distortion-corrected image.

The code of this part in the file `./im_trans.py`, function `undistort` in lines 9 through 16. This function loads the calibrated camera matrix `mtx` and distortion coeeficients `dist` (obtained and saved) from the calibration step and apply opencv function `cv2.undistort()` to a new image (taken by the same camera). The following picture shows a comparison of images before and after being undistorted. 

![alt text][image_2_1]


####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code of this part is in the file `./im_trans.py`. I tried different methods to get a binary image: including gradient, color, and region of interest. Some of them look better than others. But for robustness, I finally combined these methods. I will analyze in details in the following. 

* Gradient information.

The code is in the function `gradient_thresh()` in the file `im_trans.py` in lines 19 through 41. 

To get the gradient information, I tried Sobel filter. I first convert an input image into gray scale and apply `cv2.Sobel()`. For comparison, I plot the result of applying this filter to the same image using different directions, namely, on the x-direction, y-direction, and xy-direction (magnitude). However, the following image shows that sobel-x is good for picking up the lanes as the lanes are mostly close to vertical orientation.

![alt text][image_2_2_1]

I use the  threshold `[20,100]` to filter sobel with x-direction. The threshoulded binary image is shown below.

![alt text][image_2_2_2]

* Color information.

The code is in the function `hls_thresh()` in the  file `im_trans.py` in lines 44 through 52. 

From the lectures, HLS is shown to be good to pick up the lanes. So I convert the input image from RGB color space to HLS color space by `cv2.cvtColor()`. 
The three channels look like this 

![alt text][image_2_3_1]

It is very clear than s-channel will do a better job than others. I use  the threshold `[90,255]` to produce a binary image on s-channel. Notice that in the lecture, an exlusive - inclusive thresholding is used, i.e. `(90, 255]` while here I use a double inclusive thresholding. The result looks like this

![alt text][image_2_3_2]

The above results show that s-channel need to be complemented by gradient channel to select both sides of lanes. More comparison could be found in `output_images/binary_test*.jpg`. 

* Combination

By comining color and gradient channel, the result will show both sides of lanes.

![alt text][image_2_4_1]

Using region of interest could remove the extra pixels like sky and trees etc. in the upper part of the image. so I add the ROI selection code from previous lectures over the binary image, then I get result like this:

![alt text][image_2_4_2]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in the file `im_persp.py`. The function `get_corr_pts()` in lines 6 through 21 defines corresponding source (`src_pts`) and destination (`dst_pts`) points.These points are decided in a manual way (with the help of the region of interest in an image surround straight lanes). The source and destination points are hardcoded in the following manner:

```
    src_pts = np.array([[(imsize[0]/6+10, imsize[1]),
                         (imsize[0] / 2 - 60, imsize[1] / 2 + 100),
                         (imsize[0] / 2 + 65, imsize[1] / 2 + 100),
                         (imsize[0]*5/6 + 40, imsize[1])]],dtype=np.float32)

    dst_pts = np.array([[(imsize[0]/4,imsize[1]),(imsize[0]/4, 0),
                         (imsize[0]*3/4,0),(imsize[0]*3/4,imsize[1])]],dtype=np.float32)

```
This resulted in the following source and destination points:


| Source              | Destination   | 
|:-------------------:|:-------------:| 
|223.33332825   720.  |320.  720.     | 
|580.           460.  |320.    0.     |
|705.           460.  |960.    0.     |
|1106.66662598  720.  | 960.  720.    |


The function `warper()` in lines 28 through 33 in file `im_persp.py` computes a perspective transformation matrix and apply it to the input image.

I use an image with straight lines to tune the hardcoded corresponidng points in order to get a good perspective matrix. The warped image looks like this:

![alt text][image_2_5_1]

Then I apply the same source points and transform matrix on other images with curved lines. It turns out that the transformed lines are still parallel to each other (one example is shown below).

![alt text][image_2_5_2]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for this part is in the file `./lane_fit.py`.

Applied all techniques introduced in the previous section, the binary warped lane image looks like this:

![alt text][image_2_6_1]

I copied the sample code from the lecture to do a window searching for the lane pixels (function `poly_fit()` in `./lane_fit.py` in lines 9 through 100). The search starts from bottom of the image, with the position getting the peak value of the histogram over colume pixels. Then the searching places a rectangle centered at the base position, and counts the number of pixels inside the rectangle. When the number of pixels attains 50, the next box will be placed at the center of all these pixels, and so on so forth. When all the pixels are recognized as lane pixels, they are used to fit a 2nd-order polynomial function. The following image is an example result: green boxes are the searching boxes, all pixels inside are considered as lane pixels (red for the left lane, and blue for the right lane); the yellow curves are fitted polynomials.

![alt text][image_2_6_2]

Given the lane information from the previous frame, searching in the next frame can leverage the history. In this way tracking the lane would be easier and faster. The basic idea is to replace the searching windows by strips around fitted polynomials, then recompute the coefficients using the lane pixels. The result is shown in the following example. (Code is in the function `poly_track()` in `./lane_fit.py` in lines 103 through 161. However, this technique is not used in later processing of the video frames.)

![alt text][image_2_6_3]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I copied the code from the lecture to compute curvatures. The code is in function `curvature()` in lines 164 through 220 in my code in `./lane_fit.py`

Bacically, the curvature could be computed using the coefficients of fitted polynomial function. Based on a mapping between one pixel and distance in real world, we could compute the radius of the two lanes are:

| Left       | Right       | 
|:----------:|:-----------:| 
|2151 pixels | 1630 pixels |
|629 m       | 472 m       | 

There is a large discrepancy between the curvature of the left and the right lanes. This could be observed from the following plotting: The right lane  is dashed lines, so no as many pixels are recognized as the left lane, this could make it not so robust for the curve estimation. Another observation is the right lane containes some extra pixels at the bottom of the image which causes a bias for the polynomial estimation. Some improvment could be done for this.

![alt text][image_2_6_4]

The computation of the vehicle position is similar, using position of  `left_curverad`, `right_curverad` at the bottom of the image, I compute the difference between their center and the image center (x coordinate) which is 37.8 pixels, i.e., 0.22 m in the real world.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I copied the sample code from the lecture to plot the final result (see function `lane_plot()` in `lane_fit.py` in lines 223 through 248. The idea behind is to project fitted curve from warped image back to original image (undistorted one) 
Here is an example of my result on the test image:

![alt text][image_2_7]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The code for this part is in the file `main_lane_find.py`. The result is compute for every frame using only window searching (`poly_fit()` in `lane_fit.py`). So the processing speed is not very fast, this could be improved by applying `poly_track()`.

Here's a [link to my video result](./output_images/project_video_result.avi)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Some of the issues are discussed in previous sections. In the following I may repeat them again with some additional comments:

* In the calibration step, some manual configuration is needed to select patterns. But this is not a big issue as the calibration is needed only once. But there does exist more friendly algorithms to automatically select patterns and to even allow manual interaction to make better usage of chessboard images.

* In the beginning I thought gradient information is not robust for picking up the lanes and didnot use it. Then I found I have difficulty to recognize both sides of lanes. 

* Dashed lines will cause problem for curve fitting, this could be improved by connecting short lines like using canny method in the previous lecture. Another way is to use left lane to validate and improve robustness. Because if the curvation for the two lanes have a big difference, then the result is prone to error.

* After applying the same procedure on `challenge_video.mp4` and `harder_challenge_video.mp4`, I observed the following failure cases:

 * The road is not flat (clear separation of color in the middle of the road)
 * Cluttered shadows (such as the shadow from the trees).

	I think these two failure cases are easy to explain (it evan makes sense to have them if we look at the principle of the method), as the method basically pick up the color and gradient, so cluttered background (caused by illumination or color change) may introduce noise in lane pixels which will cause bias or even failure for the curve fitting)
	
	I think one possible way to solve this issue is to look for robust features and good ways to combine them (features which could pick up the lane but invariant to color and gradient change). Learning could also be considered. 

