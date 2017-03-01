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

[img_2_3]: ./output_images/test_perspective.jpg "Warp Example"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"


####Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---


###Camera Calibration

####1. Computed camera matrix and distortion coefficients from given chessboard images. 


The code is in the file called `./cam_calib.py`; the calibration data is in the files `/output_images/calib_pts.npz` and `/output_images/calib_mtx.npz`  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

For detecting chessboard corners, I first convert each chessboard image into gray scale, then use `cv2.findChessboardCorners()` to detect corner points. The points are plotted in the image by `cv2.drawChessboardCorners()`. The following image shows an example
of detected corners.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image_1_0]

I have found two interesting issues during finding corners: 

* The usually pattern is set to (9,5), however some times opencv fails to detect given pattern. 

An example is the above image, it gets 9X6 corners, however, opencv will simply return a false when the pattern is set to (9,5). This may due to that opencv is not very flexible for input patterns.

* When rotating the image by 90 degree (image4), the detected patterns are different
 
 
---

###Pipeline (single images)

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image_2_0]

####1. Provide an example of a distortion-corrected image.

The code of this part in the file `im_trans.py`, function `undistort` in lines 9 through 16. This function load camera matrix `mtx` and distortion coeeficients `dist` saved from calibration step and apply opencv function `cv2.undistort()` to a new image (taken by the same camera). The following picture shows a comparison before and after undistoring the image. 

![alt text][image_2_1]


####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code of this part in the file `im_trans.py`. I tried different methods to get a binary image: including gradient, color, and region of interest. Some of them look better than others. I will analyze in the following in details. 

* Gradient information.

The code is in the function `gradient_thresh()` in file `im_trans.py` in lines 19 through 41. 

To get the gradient information, I tried Sobel filter. I first convert an input image into gray scale and apply `cv2.Sobel()`. For comparison, I plot the result of this filter using different directions, namely, on the x-direction, y-direction, and xy-direction (magnitude). However, the following image shows that all of them are not ideal to pick up the lanes.

![alt text][image_2_2_1]

For example, in the lecture, it is said the x-direction is better than y-direction to pick up the lanes since the lines are mostly vertical. I use the  threshold `[20,100]` to filter sobel with x-direction. Then in the binary (below), the lanes disappear.

![alt text][image_2_2_2]

* Color information.

The code is in the function `hls_thresh()` in file `im_trans.py` in lines 44 through 52. 

From the lectures, HLS is shown to be good to pick up the lanes. So I convert the input image from RGB color space to HLS color space by `cv2.cvtColor()`. 
The three channels will look like this 

![alt text][image_2_3_1]

It is very clear than s-channel will do a better job than others. I use threshold `[90,255]` to produce a binary image on s-channel. Notice that in the lecture, an exlusive - inclusive thresholding is used, i.e. `(90, 255]]` while here I use a double inclusive thresholding. The result will look like this

![alt text][image_2_3_2]

The above results show that s-channel need to be complemented by gradient channel to select both sides of lanes. More comparison could be found in `output_images/binary_test*.jpg`. 

* Combination

By comining color and gradient channel, the result will show both sides of lanes.

![alt text][image_2_4_1]

Using region of interest could remove the extra pixels like sky and trees etc. in the upper part of the image. so I added the ROI selection code of previous lectures above the binary image, then I get results like this:

![alt text][image_2_4_2]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in the file `im_persp.py`. The function `get_corr_pts()` in lines 6 through 21 defines corresponding points of source (`src_pts`) and and destination (`dst_pts`) points in a manual way (with help of region of interest on an image containing straight lanes). The source and destination points are hardcoded in the following manner:

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


The function `warper()` in lines 28 through 33 in file `im_persp.py` compute perspectiv matrix and apply it to the input image.

I use an image with straight lines to tune the hardcoded corresponidng points in order to get a good perspective matrix. The transformed image looks like this:

![alt text][image_2_5_1]

Then I apply the same source points on other images with curved lines. It turns out that the transformed lines are still parallel to each other (one example is  shown below).

![alt text][image_2_5_2]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Applied all techniques introduced in the last section, the binary warped lane image looks like this:

![alt text][image_2_6_1]

I copied sample code from lecture to do a window searching for the lane pixels.  The search starts from bottom of the image, with the position getting the peak value of histogram for columes. Then the search place rectangles centered at the base position, and count number of pixels. When the number of pixels attains 50, the next box will placed at the center of all these pixels. And so on so forth. When all the pixels are recognized as lane pixels, they are used to fit a 2nd-order polynomial function. The result is the following: green boxes are the searched box, all pixels inside are considered as lane pixels (red for the left lane, and blue for the right lane); the yellow curves are fitted polynomial curves.

![alt text][image_2_6_2]

Given the lane information from the previous frame, searching in the next frame can leverage this information, so tracking the lane would be easier and faster. The basic idea is to replace the searching windows by strips around fitted polynomial curve, then recompute the coefficients using the lane pixels.

![alt text][image_2_6_3]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I copied the code from the lecture to compute curvatures. The code is in function `curvature()` in lines 155 through 209 in my code in `lane_fit.py`

Bacically, the curvature could be computed using the coefficients of fitted polynomial function. Based on a mapping between distance of pixels and distance in real world, we could compute the radius of the two lanes are:


| Left         | Right         | 
|:------------:|:-------------:| 
|2332.3 pixels | 1321.9 pixels |
|678.6m        | 367.6m        | 

There is a large discrepancy between the curvature  of left and right lanes. This could be observed from the following plotting: The right lane containes some  extra pixels in the beginning and this caused a bias of the polynomial estimation. Some improvment could be done for this.

![alt text][image_2_6_4]

The computation of the vehicle position is similar, using position of  left_curverad, right_curverad at the bottom of the image, I compute the difference between their center-x and the image center-x is 28.8 pixels, i.e., 0.17 m in the real world.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I copied the sample code from the lecture to plot the final result. The idea behind is to project fitted curve from warped image back to original image (undistorted one)

The code for this step is in the function `lane_plot()` 212 through 238 in my code in `lane_fit.py`.  Here is an example of my result on the test image:

![alt text][image_2_7]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

