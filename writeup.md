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

[image1]: ./output_images/cam_calib.png "Camera calibration"
[img_test]: ./test_images/straight_lines2.jpg "Road Transformed"
[img_undistort_lane]: ./output_images/undistort_straight_lines2.jpg "Undistort"
[img_binary1]: ./output_images/binary_straight_lines2.jpg "Binary1"
[img_binary2]: ./output_images/binary_test1.jpg "Binary2"
[img_binary3]: ./output_images/binary_combine.jpg "Binary3"
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

![alt text][image1]

I have found two interesting issues during finding corners: 

* The usually pattern is set to (9,5), however some times opencv fails to detect given pattern. 

An example is the above image, it gets 9X6 corners, however, opencv will simply return a false when the pattern is set to (9,5). This may due to that opencv is not very flexible for input patterns.

* When rotating the image by 90 degree (image4), the detected patterns are different
 
 
---

###Pipeline (single images)

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][img_test]
####1. Provide an example of a distortion-corrected image.

After the camera calibration, I get camera matrix `mtx`, and distortion coeeficients `dist`, applying opencv function `cv2.undistort()` to new images taken by the same camera could get undistorted image. The result is 

![alt text][img_undistort_lane]


####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code of this part in the file `image_trans.py`, including image distortion from calibration matrix, image transformation to gray color or to different color space and generating binary images.

To use the gradient threshold, I first convert the image into gray scale and apply `cv2.Sobel()` on the x-direction since x-direction is better than y-direction to pick the lanes. Then threshold `[20,100]` is used to produce a binary image thresholded by gradient (see the middle figure of the above picture.

I also use color to threshold an image. First I convert the image from RGB color space to HLS color space by `cv2.cvtColor`. And S-channel image is chosen to do a thresholding. The parameter is set to `[90,255]`. Notice that in the lecture, an exlusive - inclusive thresholding is used, i.e. `(90, 255]]` while here I use a double inclusive thresholding. 

![alt text][img_binary1]

By trying different test images, I found that s-channel image is relatively robust while gradient is not so robust under different conditions. For example, see the following picture.

![alt text][img_binary2]

I think a combination of color and region of interest will be able to better select the lanes, so I added the ROI selection code of previous lectures above the color selection, then I get results like this:

![alt text][img_binary3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 23 through 28 in the file `./trans_perspective.py`. The `warper()` function takes as inputs an image (`img`), and compute source (`src_pts`) as well as and destination (`dst_pts`) points.  I chose the hardcode the source and destination points in the following manner:

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

