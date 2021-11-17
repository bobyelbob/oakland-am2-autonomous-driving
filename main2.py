import cv2
import numpy as np
import time
import math
# import motor_controller as mc
import pyfirmata

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 320) # set width to 320 px
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240) # set height to 240 px

height = 240
width = 320
frame_size = (320,240)
# Initialize video writer object
output = cv2.VideoWriter('./output_video.mp4', 
                         cv2.VideoWriter_fourcc(*'XVID'), 
                         20, frame_size)


# vertices = np.array([[[0,240-65], [110,96], [320-110,96],
#                     [320, 240-65]]], dtype=np.int32)
vertices = np.array([[[0,240],[0,240-65], [110,120], [320-110,120],
                    [320, 240-65],[320, 240]]], dtype=np.int32)

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
    # Defining a blank mask to start with
    mask=np.zeros_like(frame)
    
    # Defining a three channel or one channel color to fill the mask with 
    # depending on the input image
    
    if len(frame.shape) > 2:
        channel_count = frame.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    # Filling pixels inside the polygon defined in 'vertices' with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    # Returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(frame, mask)
    return masked_image

def region_of_interest_2(frame, vertices):
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
                                    np.array([]), minLineLength=30, maxLineGap=10)
    return line_segments

def average_slope_intercept(frame, line_segments):
    lane_lines = []

    if line_segments is None:
        print("no line segment detected")
        return lane_lines
    
    # height, width,_ = frame.shape
    height = 240
    width = 320
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
    # height, width, _ = frame.shape
    height = 240
    width = 320
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

    # height, width, _ = frame.shape
    height = 240
    width = 320

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
        return 1000

    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
    print(angle_to_mid_deg)
    steering_angle = angle_to_mid_deg + 90
    print(steering_angle)

    return steering_angle


def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5):

    heading_image = np.zeros_like(frame)
    # height, width, _ = frame.shape
    height = 240
    width = 320


    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)

    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image   

def process_frame(frame):
    gray = grayscale(frame)
    blur = gaussian_blur(gray)
    cany = canny(blur)
    # vertices = 
    # result = region_of_interest(cany, vertices)

    rho = 2
    theta = np.pi / 180 * 1
    threshold = 10
    min_line_len = 20
    max_line_gap = 10
    frm = result
    lines = cv2.HoughLinesP(frm, rho, theta, threshold, np.array([]),
                            min_line_len, max_line_gap)
    # lines = process_lines(lines, frame)
    line_frame = np.zeros((320, 240, 3), dtype=np.uint8)
    draw_lines(line_frame, lines, thickness = 8)
    result = weighted_img(line_image, frame)

    return result

def mapping(x, in_min, in_max, out_min, out_max):
    # returs mapped value according to in and out values
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;

def right(steering):
    steer = mapping(steering, 100, 180, 0, 1)
    enA.write(steer)
    enB.write(1 - steer)
#     board.digital[in1].write(1)
#     board.digital[in2].write(0)
#     board.digital[in3].write(1)
#     board.digital[in4].write(0)
    
def left(steering):
    steer = mapping(steering, 0, 80, 0, 1)
    enA.write(1 - steer)
    enB.write(steer)
    # board.digital[in1].write(0)
    # board.digital[in2].write(1)
    # board.digital[in3].write(0)
    # board.digital[in4].write(1)
    
def reverse():
    board.digital[in1].write(0)
    board.digital[in2].write(1)
    board.digital[in3].write(1)
    board.digital[in4].write(0)
        
def forward():
    board.digital[in1].write(1)
    board.digital[in2].write(0)
    board.digital[in3].write(0)
    board.digital[in4].write(1)

def stop():
    enA.write(0)
    enB.write(0)

def run():
    enA.write(1)
    enB.write(1)

board = pyfirmata.Arduino('/dev/ttyUSB0')

led = 13
in1 = 9
in2 = 8
in3 = 3
in4 = 4
enA = board.digital[10]
enA.mode = pyfirmata.PWM
enB = board.digital[5]
enB.mode = pyfirmata.PWM

def drive(steering_angle):
    if steering_angle > 100:
        # need to turn right
        right(steering_angle)
    elif steering_angle < 80
        # need to turn left
        left(steering_angle)
    elif 80 <= steering_angle <= 100:
        forward()
        run()
    elif steering_angle == 1000:
        #lane line not detected
        stop()
      
    # if 80 <= steering_angle < 90:
    #     forward()
    #     run()
    #     print('forward')
    # elif 90 < steering_angle <= 100:
    #     forward()
    #     run()
    #     print('forward')
        # if 80 <= steering_angle <= 100:
        #     forward()
        #     run()
        #     print('forward')
        # elif steering_angle < 80:
        #     left()
        #     run()
        #     print('left')
        # elif steering_angle > 100:
        #     right()
        #     run()
        #     print('right')
    # elif steering_angle == 90:
    #     stop()
    #     print(stop)
            # else:
            #     stop()


while True:
    
    ret, frame = video.read()
    # print(frame.dtype)
    # frame = cv2.flip(frame,-1) # flip image vertically
    if ret:
        
        original_frame = frame.copy()
        gray_frame = grayscale(frame)
        blur_frame = gaussian_blur(gray_frame, 5)
        canny_frame = canny(blur_frame, 80, 240)
        roi_frame = region_of_interest(canny_frame, vertices)
        line_segments = detect_line_segments(roi_frame)
        lane_lines = average_slope_intercept(frame, line_segments)
        lane_lines_image = display_lines(original_frame, lane_lines)
        steering_angle = get_steering_angle(original_frame, lane_lines)
        heading_image = display_heading_line(lane_lines_image, steering_angle)

        cv2.imshow('gray', gray_frame)
        cv2.imshow('canny', canny_frame)
        cv2.imshow('roi', roi_frame)
        cv2.imshow('processed', heading_image)
        output.write(heading_image)
        # cv2.imwrite('original.jpg', canny_frame)

        run()
        drive(steering_angle)
        
        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        break
    
video.release()
output.release()
cv2.destroyAllWindows()
stop()