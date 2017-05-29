# **Finding Lane Lines on the Road** 

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Main Pipeline

The main function is process_image(image), 
and it is the entry point for the lane lines detection process.
It receives a color (3 channel, 8 bit per channel, rgb) image 
and returns the same image with the lane line markers added.
In this project red was used for the left line and green for the right one.

The process steps are the following:

    1) Turn Image to Gray Scale single channel
    test_image = grayscale(image)

    #   2) Mask the region of interest (set everything else to black)
    vertices = get_poly_of_interest(test_image)
    test_image = region_of_interest(test_image, vertices)

    #   3) Apply some blur (Gaussian) to the previous image so make it softer
    kernel_size = 5 # Must be an odd number (3, 5, 7...)
    test_image = cv2.GaussianBlur(test_image,(kernel_size, kernel_size),0)

    #   4) Apply Canny algorithm to get borders gradient
    test_image = canny(test_image, low_threshold, high_threshold)

    #   5) Apply Hough transformation to get lines
    hough_lines(test_image, rho, theta, threshold, min_line_length, max_line_gap, image)

The last step returns the original color image with the lane lines added. 
The hough_lines function get the candidate lines and call draw_lines(), 
which has the following prototype:
def def draw_lines(img, lines, color=[255, 0, 0], thickness=4):

This function bascially does the following:
	1) Get the center x coordinate of the region of interest
	2) Get the top and botton of the region of interest as well as 
	   its inner margins (defined in the same function as process parameters)
	3) Walk through all the lines, discard horizontal and vertical lines,
	   and split them in left and right candidate lines 
	   based on the center of the area of interest and calculate
	   the top left, and bottom left points of the left lane line as well as the right
	   pair of points based on coordinates within the margines of the area of interest and 
	   based on max values of x for the left line and min values of x for the right line
	
	The result of this subprocess is something like this:

	-------------------------------------------------------------------------------

	                                x                x 
	              (top_left_x1, top_left_y1)  (top_right_x1, top_right_y1)
				                 

               x                                                             x 
        (bottom_left_x1, bottm_left_y1)            (bottom_right_x1, bottm_right_y1)

	-------------------------------------------------------------------------------

	4) Based on those values the slope (m) is calculated for left and right lines
	5) On both cases there is a check for infinite slope condition 
	   and for anormal slope value based on defined limits
	6) Then b is calculated (b = y - m*x)
	7) Both top and bottom y coordinates of the region of interest are used 
	   to extrapolate the lines
	8) Lines are added to the original image

	Important: This function includes a boolean show_aux_markers flag which 
	enables handy additional markers to debug / fine tune the process.

[image1]: ./CarND_LL_P1/test_images/solidYellowLeft.jpg_processed.png
[image2]: ./CarND_LL_P1/test_images_output/solidYellowLeft.jpg_processed.png

### 2. Potential shortcomings with this pipeline

There are limitations in the draw_lines function
which I would like to be more stable and accurate.

### 3. Pipeline improvements

Add more in deph analysis to the points in the draw_lines function
to minimize error, for example get the line based on an average of 
all nearest points.
