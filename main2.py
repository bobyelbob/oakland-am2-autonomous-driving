import cv2
import numpy as np
import time
import math

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 320) # set width to 320 px
video.set(cv2.CAP_PROP_FRAME_HEIGHT,240) # set height to 240 px

def grayscale(frame):
    # Applies grayscale transorm to an image
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def gaussian_blur(frame, kernel_size):
    # Applies gaussian noise kernel
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

def canny(frame, low_threshold, high_threshold):
    # Applies canny transform to an image
    return cv2.Canny(frame, low_threshold, high_threshold)

def region_of_interest(frame, vertices):
    # Defining a region of interest mask
    mask=np.zeros_like(frame)
    
    if len(frame.shape) > 2:
        channel_count = frame.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    masked_image = cv2.bitwise_and(frame, mask)
    return masked_image

def draw_lines(frame, lines, color=[255,0,0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(frame, (x1,y1), (x2,y2), color, thickness)
    
def hough_lines(frame, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(frame, rho, theta, threshold, np.array([]),
                            minLineLenght=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(frame, initial_img, α=0.9, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)


def detect_line_segments(cropped_edges):
    rho = 1  
    theta = np.pi / 180  
    min_threshold = 10 
    line_segments = cv2.HoughLinesP(cropped_edges, rho, theta, min_threshold, 
                                    np.array([]), minLineLength=5, maxLineGap=0)
    return line_segments

def average_slope_intercept(frame, line_segments):
    lane_lines = []

    if line_segments is None:
        print("no line segment detected")
        return lane_lines

    height, width,_ = frame.shape
    left_fit = []
    right_fit = []
    boundary = 1/3

    left_region_boundary = width * (1 - boundary) 
    right_region_boundary = width * boundary 

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                print("skipping vertical lines (slope = infinity)")
                continue

            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)

            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    # lane_lines is a 2-D array consisting the coordinates of the right and left lane lines
    # for example: lane_lines = [[x1,y1,x2,y2],[x1,y1,x2,y2]]
    # where the left array is for left lane and the right array is for right lane 
    # all coordinate points are in pixels
    return lane_lines

def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 / 2)  # make points from middle of the frame down

    if slope == 0: 
        slope = 0.1    

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return [[x1, y1, x2, y2]]

def display_lines(frame, lines, line_color=(0, 255, 0), line_width=6): # line color (B,G,R)
    line_image = np.zeros_like(frame)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)

    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)  
    return line_image


def get_steering_angle(frame, lane_lines):

    height, width, _ = frame.shape

    if len(lane_lines) == 2: # if two lane lines are detected
        _, _, left_x2, _ = lane_lines[0][0] # extract left x2 from lane_lines array
        _, _, right_x2, _ = lane_lines[1][0] # extract right x2 from lane_lines array
        mid = int(width / 2)
        x_offset = (left_x2 + right_x2) / 2 - mid
        y_offset = int(height / 2)  

    elif len(lane_lines) == 1: # if only one line is detected
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
        y_offset = int(height / 2)

    elif len(lane_lines) == 0: # if no line is detected
        x_offset = 0
        y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  
    steering_angle = angle_to_mid_deg + 90 

    return steering_angle


def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5):

    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)

    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image    

while True:

    ret, frame = video.read()
    # frame = cv2.flip(frame,-1) # flip image vertically

    gray_frame = grayscale(frame)
    blur_frame = gaussian_blur(gray_frame, 5)
    canny_frame = canny(blur_frame, 50, 150)
    cv2.imshow('original', canny_frame)
    # cv2.imwrite('original.jpg', frame)


    key = cv2.waitKey(1)
    if key == 27:
        break
    
video.release()
cv2.destroyAllWindows()