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
[image_2_0]: ./test_images/test1.jpg "Road Transformed"
[image_2_1]: ./output_images/test_undistort.png "Undistort"
[image_2_2_1]: ./output_images/test_sobel.png "gradient"
[image_2_2_2]: ./output_images/test_sobel_binary.png "gradient"
[image_2_3_1]: ./output_images/test_hls.png "color"
[image_2_3_2]: ./output_images/test_s_binary.png "color"
[image_2_4]: ./output_images/test_combine.png "combine"
[image_2_5]: ./output_images/test_persp.png "combine"

[image6]: ./output_images/binary_combine.png "Binary3"
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

By trying different test images, I found that s-channel image is more robust than the gradient. More comparison could be found in `output_images/binary_test*.jpg`. 

* Combination

I think a combination of color and region of interest will be able to better select the lanes, espetially, region of interest will help to remove the sky and tree part in the upper part of the image. so I added the ROI selection code of previous lectures above the color selection, then I get results like this:

![alt text][image_2_4]

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


The function `warper()` in lines 23 through 28 compute perspectiv matrix and apply it to the input image.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image (shown as follows).

![alt text][img_2_3]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

