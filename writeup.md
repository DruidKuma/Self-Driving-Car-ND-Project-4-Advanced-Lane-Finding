# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./test_images/straight_lines1.jpg "Test Image"
[image2]: ./output_images/camera_cal/corners_found5.jpg "Corners Example"
[image3]: ./output_images/chess_dist.jpg "Chess Distorted"
[image4]: ./output_images/chess_undist.jpg "Chess Undistorted"
[image5]: ./output_images/test_undist.jpg "Test Undistorted"
[image6]: ./output_images/s_channel_problem.png "S channel problem"
[image7]: ./output_images/s_channel_binary.png "S channel binary"
[image8]: ./output_images/h_channel_binary.png "H channel binary"
[image9]: ./output_images/r_channel_binary.png "R channel binary"
[image10]: ./output_images/color_thresh_combined.png "Combined Color Threshold"
[image11]: ./output_images/sobel_x_binary.png "Sobel X Threshold"
[image12]: ./output_images/sobel_y_binary.png "Sobel Y Threshold"
[image13]: ./output_images/grad_mag_binary.png "Gradient Magnitude Threshold"
[image14]: ./output_images/grad_dir_binary.png "Gradient Direction Threshold"
[image15]: ./output_images/grad_combined_binary.png "Combined Gradient Threshold"
[image16]: ./output_images/thresh_final.png "Final Threshold Result"
[image17]: ./output_images/warped_result.png "Warped Result"
[image18]: ./output_images/lines_detected.png "Line Detection Result"
[image19]: ./output_images/final_result.png "Final Result"

[video1]: ./project_result_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step located in file `project_code.py` (lines 18-51).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (9, 6) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.
While detecting corners on the chessboard images, I draw detected corners to see everything goes right (all other images with corners can be found in output_images/camera_cal folder):

![][image2]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![][image3]
![][image4]

I calculate distortion coefficients and matrix only once and save them into a file `calibration_result_pickle.p` for further use.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Using saved matrix and distortion coefficients, obtained from previous step, I applied `cv2.undistort()` to get the following result on the test image:

![][image1]
![][image5]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image.

Firstly, I applied color thresholding to image. I chose converting an image to HSV to obtain H and S channels. S channel does pretty good job, but the problem with it is that it captures shadows, and it spoiled line detection:

![][image6]

H-channel appeared to be a good option to exlude shadows from S-channel. In addition to this pair I took R-channel from original RGB image which is good in capturing white lines.

After playing a bit with threshold boundaries, I obtained the following results from color thresholding:

**S-channel:**
![][image7]

**H-channel:**
![][image8]

**R-channel:**
![][image9]

**Combined:**
![][image10]

The code for color thresholding can be found in file `project_code.py` (lines 246-259)

After this I also applied gradient thresholding
I combined Sobel X with Sobel Y binary masks and gradient magnitude with gradient direction binary masks to obtain the following results:

**Sobel X:**
![][image11]

**Sobel Y:**
![][image12]

**Graient Magnitude:**
![][image13]

**Graient Direction:**
![][image14]

**Combined:**
![][image15]

The code for gradient thresholding can be found in `project_code.py` file on lines 262-299.

After these steps I combined color thresholding and gradient thresholding altogether and got the final result (lines 302-309):

![][image16]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `unwarp()`, which appears in lines 58 through 74 in the file `project_code.py`. I chose to hardcode the source and destination points so that they fit the following mapping:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 490, 482      | 0, 0        | 
| 810, 482      | 1280, 0      |
| 1250, 720     | 1250, 720      |
| 40, 720      | 40, 720        |

So, as could be seen from the point mapping, I left the bottom corners unwarped (mapped as is), and the two top points (defining the lanes pretty far from the car) are mapped to the top corners of the warped image.
After applying such birds-eye effect on the above thresholded binary picture, I got the following result:

![][image17]

I also calculated the inverse matrix for unwarping the image with detected and colored lane.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Firstly I took a histogram along all the column in the lower half of the image to identify peaks (likely columns with lines) (lines 79-86). Then, starting from peaks positions, I use sliding windows, placed around the line centers to follow the lines up to the top of the frame and find pixels corresponding to left and right lines separately (lines 89-144). Last, but not least, I fitted the second order polynomials to each of lines (lines 148-149). As a result, I got the following:

![][image18]

I used global variables to apply sliding windows only once and slightly hurry up the processing of video stream (lines 154-155, 157-161)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this on lines 188-197 on `project_code.py` file, measuring the average of the two curves to draw them on video, and on lines 200-207, calculating the position of car with respect to center.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The final drawing of the results is located on lines 211-237 of the same `project_code.py` file.

After doing all above steps I came up with the following result:

![][image19]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [Video Result][video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I faced a lot of issues with image preprocessing techniques. When I started the project, I suffered from thresholding either captures too much data (shadows) or don't capture lines at all (like in very light places). Only after endless trials I came up with a solution which satisfied me.
 
Actually, the described pipeline works pretty well on the required project video, but fails on the challenges. I see a lot of places for further improving, e.g.: 

* Not hardcode, but compute points of region of interest to be able to successfully detect lane on curvy road
* Improve lane line points detection (as global variables are not a good code style)
* Improve curvature measuring to satisfy the norms and compute average curvature along certain number of frames instead of for every frame separately
* Implement proper tracking of line detection, best fit etc.
