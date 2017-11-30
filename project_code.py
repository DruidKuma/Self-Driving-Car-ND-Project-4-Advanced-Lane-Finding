import pickle
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

def showImage(img):
	plt.imshow(img, cmap='gray')
	plt.show()


def saveImage(img, name):
    cv2.imwrite(name, img)


def generateCameraCalibrationMatrix(img_width=1280, img_height=720):
	#dimensions of chessboard on calibration images
	grid_w = 9 #width
	grid_h = 6 #height

	images = glob.glob('camera_cal/calibration*.jpg')

	objpoints = [] # 3D points in real world space
	imgpoints = [] # 2D points in image plane

	#object points for undistortion like (0,0,0), (1,0,0) up to (8,5,0)
	objp = np.zeros((grid_h * grid_w, 3), np.float32)
	objp[:,:2] = np.mgrid[0:grid_w, 0:grid_h].T.reshape(-1,2)

	for fname in images:
	    img = cv2.imread(fname)
	    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	    # Find the chessboard corners
	    ret, corners = cv2.findChessboardCorners(gray, (grid_w,grid_h), None)

	    # If found, add object points, image points
	    if ret == True:
	        objpoints.append(objp)
	        imgpoints.append(corners)

	#calculate calibration matrix
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (img_width, img_height), None, None)

	# Save the camera calibration result for later use
	dist_pickle = {}
	dist_pickle["mtx"] = mtx
	dist_pickle["dist"] = dist
	pickle.dump(dist_pickle, open( "calibration_result_pickle.p", "wb" ))


def undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


def unwarp(img, img_width=1280, img_height=720):
    # Approximately for corner points of the region of interest
    src = np.float32([[490, 482],[810, 482],
                      [1250, 720],[40, 720]])

    # Approximate target points
    dst = np.float32([[0, 0], [1280, 0], 
                     [1250, 720],[40, 720]])
    
    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Calculate the inverse perspective transform matrix
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    # Warp the image
    return cv2.warpPerspective(img, M, (img_width, img_height)), Minv


def findLeftRightFit(binary_warped):
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
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
	    win_y_low = binary_warped.shape[0] - (window+1)*window_height
	    win_y_high = binary_warped.shape[0] - window*window_height
	    win_xleft_low = leftx_current - margin
	    win_xleft_high = leftx_current + margin
	    win_xright_low = rightx_current - margin
	    win_xright_high = rightx_current + margin
	    # Draw the windows on the visualization image
	    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
	    (0,255,0), 2) 
	    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
	    (0,255,0), 2) 
	    # Identify the nonzero pixels in x and y within the window
	    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
	    (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
	    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
	    (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
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

	return left_fit, right_fit


left_fit = []
right_fit = []
def detectLines(binary_warped):
	global left_fit
	global right_fit

	if left_fit == [] or right_fit == []:
		left_fit, right_fit = findLeftRightFit(binary_warped)

	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	margin = 100
	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
	left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
	left_fit[1]*nonzeroy + left_fit[2] + margin))) 

	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
	right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
	right_fit[1]*nonzeroy + right_fit[2] + margin)))  

	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]
	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Measure Radius of Curvature for each lane line
	ym_per_pix = 30./720 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meteres per pixel in x dimension

	y_eval = np.max(ploty)
	left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
	avg_curverad = int((left_curverad + right_curverad)/2)

	#Calculate distance from center
	image_center = binary_warped.shape[1]/2
	image_height = binary_warped.shape[0]
	car_position = image_center
	#Find distance between car position and middle of left and right x_intercept
	left_fit_x_intercept = left_fit[0]*image_height**2 + left_fit[1]*image_height + left_fit[2]
	right_fit_x_intercept = right_fit[0]*image_height**2 + right_fit[1]*image_height + right_fit[2]
	center_dist = xm_per_pix * (car_position - (left_fit_x_intercept + right_fit_x_intercept) / 2)

	return ploty, left_fitx, right_fitx, avg_curverad, center_dist

def drawLane(img, warped, Minv, ploty, left_fitx, right_fitx, avg_curverad, center_dist):
	# Create an image to draw the lines on
	warp_zero = np.zeros_like(warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
	# Combine the result with the original image
	result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

	# Print radius of curvature and distance from center on video
	cv2.putText(result, 'Curve Radius {} m'.format(avg_curverad), (50,50),
	         fontFace = 16, fontScale = 1.2, color=(255,255,255), thickness = 2)

	direction = 'right' if center_dist > 0 else 'left'
	cv2.putText(result, 'Distance from center {:04.3f} m to the {}'.format(abs(center_dist), direction), (50,100),
	         fontFace = 16, fontScale = 1.2, color=(255,255,255), thickness = 2)

	return result


def thresholdChannel(channel, thresh=(0,255)):
	result = np.zeros(channel.shape, dtype=np.uint8)
	result[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
	return result


def colorThreshold(img):
	hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

	s_binary = thresholdChannel(hls[:,:,2], thresh=(90,255))
	h_binary = thresholdChannel(hls[:,:,0], thresh=(15,100))
	r_binary = thresholdChannel(img[:,:,0], thresh=(200,255))

	combined_binary = np.zeros_like(s_binary)

	# H-chanel is used to exclude shadows, captured by S-channel.
	# R-channel well captures white lines and forms a good union with S
	combined_binary[((s_binary == 1) & (h_binary == 1)) | (r_binary == 1)] = 1

	return combined_binary


def gradientThreshold(img):
	# Convert to gray
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Compute Sobel for bith X and Y
	sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
	sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

	absSobelX = np.absolute(sobelX)
	absSobelY = np.absolute(sobelY)

	# Form threshold binaries from Sobels
	# Convert to 8 bit
	scaledSobelX = np.uint8(255 * absSobelX / np.max(absSobelX))
	scaledSobelY = np.uint8(255 * absSobelY / np.max(absSobelY))

	x_binary = np.zeros(scaledSobelX.shape, dtype=np.uint8)
	x_binary[(scaledSobelX >= 35) & (scaledSobelX <= 200)] = 1

	y_binary = np.zeros(scaledSobelY.shape, dtype=np.uint8)
	y_binary[(scaledSobelY >= 35) & (scaledSobelY <= 200)] = 1

	# Form threshold Sobel magnitude binary
	sobelMagnitude = np.sqrt(np.add(np.square(sobelX), np.square(sobelY)))
	scaledSobelMagnitude = np.uint8(255 * sobelMagnitude / np.max(sobelMagnitude))
	mag_binary = np.zeros(scaledSobelMagnitude.shape, dtype=np.uint8)
	mag_binary[(scaledSobelMagnitude >= 50) & (scaledSobelMagnitude <= 250)] = 1

	# Form threshold Direction Sobel binary
	gradientDirection = np.arctan2(absSobelY, absSobelX)
	dir_binary = np.zeros(gradientDirection.shape, dtype=np.uint8)
	dir_binary[(gradientDirection > 0.7) & (gradientDirection < 1.3)] = 1

	# Combine all gradients
	combinedGradients = np.zeros_like(dir_binary)
	combinedGradients[((x_binary == 1) & (y_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

	return combinedGradients


def thresholdPipeline(img):
	color_binary = colorThreshold(img)
	gradient_binary = gradientThreshold(img)

	combined = np.zeros_like(color_binary)
	combined[(gradient_binary == 1) | (color_binary == 1)] = 1

	return combined


def wholePipeline(img):
	dist_pickle = pickle.load( open( "calibration_result_pickle.p", "rb" ) )
	mtx = dist_pickle["mtx"]
	dist = dist_pickle["dist"]

	undistorted = undistort(img, mtx, dist)
	thresholded = thresholdPipeline(undistorted)
	unwarped, Minv = unwarp(thresholded)

	ploty, left_fitx, right_fitx, avg_curverad, center_dist = detectLines(unwarped)

	result = drawLane(undistorted, unwarped, Minv, ploty, left_fitx, right_fitx, avg_curverad, center_dist)
	return result


if __name__ == '__main__':
	#generateCameraCalibrationMatrix()

	# image = cv2.imread('test_images/straight_lines1.jpg')
	# result = wholePipeline(image)
	# showImage(result)

	clip1 = VideoFileClip("project_video.mp4")
	white_clip = clip1.fl_image(wholePipeline)
	white_clip.write_videofile('project_result_video.mp4', audio=False)










