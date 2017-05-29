#imports section
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import cv2
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML


#Basic Flow:
#a) Read media (image or video)
#b) Process media
#   1) Turn Image to Gray Scale single channel
#   2) Mask the region of interest (set everything else to black)
#   3) Apply some blur (Gaussian) to the previous image so make it softer
#   4) Apply Canny algorithm to get borders gradient
#   5) Apply Hough transformation to get lines
#c) Output media (image or video with the resulting lines on top)

#functions section
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
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
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def get_top_of_interest(img):
    """
    `img` the image to be processed.
        
    Returns the y coordinate of the region of interest.
    """
    return img.shape[0] * 0.60

def get_bottom_of_interest(img):
    """
    `img` the image to be processed.
        
    Returns the y coordinate of the region of interest.
    """
    return img.shape[0]

def get_poly_of_interest(img):
    """
    `img` the image to be processed.
        
    Returns the vertices of a poly based on image size.
    """

    height = img.shape[0]
    width = img.shape[1]

    top = get_top_of_interest(img)
    top_width = 0.052 * width
    top_left_x = width / 2 - width * 0.0125 - top_width / 2
    top_right_x = width / 2 + top_width / 2
    bottom_y = height * 0.9
    vertices = np.array([[(0,bottom_y),(top_left_x, top), (top_right_x, top), (width,bottom_y)]], dtype=np.int32)
    return vertices

def get_horiz_center_of_interest(img):
    """
    `img` the image to be processed.
        
    Returns the x coord of the center of the area of interest.
    """

    vertices = get_poly_of_interest(img)
    top_left_x = vertices[0][1][0]
    top_right_x = vertices[0][2][0]

    return (top_left_x + top_right_x) / 2

def draw_lines(img, lines, color=[255, 0, 0], thickness=4):
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
    
    #The center of the region of interest is used to split
    #left and right lane lines
    center_x = get_horiz_center_of_interest(img)
    
    #These y coordintates are handy to extrapolate lane lines
    y_top_of_interest = get_top_of_interest(img)
    y_bottom_of_interest = get_bottom_of_interest(img)

    min_left_m = 1000000
    max_right_m = -1000000

    for line in lines:
        for x1,y1,x2,y2 in line:
            #avoid horizontal lines
            if y1 != y2 and x2 != x1:
                m = (y2-y1)/(x2-x1)

                #left lines
                if x1 < center_x:
                    if m < min_left_m:
                        min_left_m = m

                #right lines
                if x1 > center_x:
                    if m > max_right_m:
                        max_right_m = m

    #Write m of lane lines for all the frames for further analysis
    white_output = 'test_videos_output/'
    f = open( white_output + 'lane_m_values.txt', 'a' )
    f.write( str(min_left_m) + "," + str(max_right_m) + '\n' )
    f.close()

    for line in lines:
        for x1,y1,x2,y2 in line:
            #avoid horizontal lines
            if y1 != y2 and x2 != x1:
                m = (y2-y1)/(x2-x1)
                b = y2 - m * x2
                x_top = int((y_top_of_interest - b) / m)
                x_bottom = int((y_bottom_of_interest - b) / m)
                #left lines
                if m == min_left_m:
                    cv2.line(img, (x_top, int(y_top_of_interest)), (x_bottom, int(y_bottom_of_interest)), [255, 0, 0], thickness)

                #right lines
                if m == max_right_m:
                    cv2.line(img, (x_top, int(y_top_of_interest)), (x_bottom, int(y_bottom_of_interest)), [0, 255, 0], thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, original_img):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    draw_lines(original_img, lines)
    return original_img

# Python 3 has support for cool math symbols.

#def weighted_img(img, initial_img, ?=0.8, ?=1., ?=0.):
#    """
#    `img` is the output of the hough_lines(), An image with lines drawn on it.
#    Should be a blank image (all black) with lines drawn on it.
    
#    `initial_img` should be the image before any processing.
    
#    The result image is computed as follows:
    
#    initial_img * ? + img * ? + ?
#    NOTE: initial_img and img must be the same shape!
#    """
#    return cv2.addWeighted(initial_img, ?, img, ?, ?)


def process_image(image):
    #process image section

    #b) Process media
    #   1) Turn Image to Gray Scale single channel
    test_image = grayscale(image)

    #   2) Mask the region of interest (set everything else to black)
    vertices = get_poly_of_interest(test_image)
    test_image = region_of_interest(test_image, vertices)

    #   3) Apply some blur (Gaussian) to the previous image so make it softer
    kernel_size = 5 # Must be an odd number (3, 5, 7...)
    test_image = cv2.GaussianBlur(test_image,(kernel_size, kernel_size),0)

    #   4) Apply Canny algorithm to get borders gradient
    mask = np.zeros_like(test_image)
    low_threshold = 1
    high_threshold = 180
    test_image = canny(test_image, low_threshold, high_threshold)

    #   5) Apply Hough transformation to get lines
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40 #minimum number of pixels making up a line
    max_line_gap = 30    # maximum gap in pixels between connectable line segments
    line_image = np.copy(test_image)*0 # creating a blank to draw lines on

    return hough_lines(test_image, rho, theta, threshold, min_line_length, max_line_gap, image)


#Main loop for images *************************************************************************
images_path = "test_images/"
output_images_path = "test_images_output/"
for image_name in os.listdir(images_path):
    
    #a) Read media (image or video)
    image = mpimg.imread(images_path + image_name)
    processed_image = process_image(image)
    mpimg.imsave(output_images_path + image_name + "_processed.png", processed_image)

##Main loop for videos *************************************************************************
videos_path = "test_videos/"
white_output = 'test_videos_output/'
for video_name in os.listdir(videos_path):

    clip = VideoFileClip(videos_path + video_name).subclip(0,2)
    #clip = VideoFileClip(videos_path + video_name)
    white_clip = clip.fl_image(process_image)
    white_clip.write_videofile(white_output + video_name, audio=False)
    pass

