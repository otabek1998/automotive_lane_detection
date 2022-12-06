import numpy as np
import cv2

capture = cv2.VideoCapture("line_video.mp4")

def canny_func(frame):
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    grey = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    canny_frame = cv2.Canny(grey, 50, 130)
    return canny_frame

def do_mask(frame):
    height = frame.shape[0]
    polygon = np.array([
        [(100, height), (700, height), (400 ,400)]
    ])
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, polygon, 255)
    region = cv2.bitwise_and(frame, mask)
    return region
def calculate_lines(frame, lines):
    left = []
    right = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            # Fits a linear polynomial to the x and y coordinates and returns a vector of coefficients which describe the slope and y-intercept
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            y_intercept = parameters[1]
            # If slope is negative, the line is to the left of the lane, and otherwise, the line is to the right of the lane
            if slope < 0:
                left.append((slope, y_intercept))
            else:
                right.append((slope, y_intercept))
        # Averages out all the values for left and right into a single slope and y-intercept value for each line
    left_avg = np.average(left, axis = 0)
    right_avg = np.average(right, axis = 0)
    # Calculates the x1, y1, x2, y2 coordinates for the left and right lines
    left_line = calculate_coordinates(frame, left_avg)
    right_line = calculate_coordinates(frame, right_avg)
    return np.array([left_line, right_line])

def calculate_coordinates(frame, parameters):
    try:
        slope, intercept = parameters
    except TypeError:
        slope, intercept = 0.001, 0
    #slope, intercept = parameters
    # Sets initial y-coordinate as height from top down (bottom of the frame)
    y1 = frame.shape[0]
    # Sets final y-coordinate as 150 above the bottom of the frame
    y2 = int(y1 - 150)
    # Sets initial x-coordinate as (y1 - b) / m since y1 = mx1 + b
    x1 = int((y1 - intercept) / slope)
    # Sets final x-coordinate as (y2 - b) / m since y2 = mx2 + b
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def display_lines(frame, lines):
    # Creates an image filled with zero intensities with the same dimensions as the frame
    lines_display = np.zeros_like(frame)
    # Checks if any lines are detected
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # Draws lines between two coordinates with green color and 5 thickness
            cv2.line(lines_display, (x1, y1), (x2, y2), (255, 255, 0), 5)
    return lines_display


def turn_predict(image_center, right_lane_pos, left_lane_pos):
    lane_center = left_lane_pos + (right_lane_pos - left_lane_pos) / 2

    if (lane_center - image_center < -20):
        return ("Turning left")
    elif (lane_center - image_center < 10):
        return ("straight")
    else:
        return ("Turning right")

out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 45, (800,600))

while (capture.isOpened()):
    ret, frame = capture.read()
    h, w, d = frame.shape
    frame= cv2.resize(frame, (800, 600))
    timer = cv2.getTickCount()
    frame_new = canny_func(frame)
    frame_new = do_mask(frame_new)
    hough = cv2.HoughLinesP(frame_new, 2, np.pi / 180, 100, np.array([]), minLineLength=100, maxLineGap=50)
    lines = calculate_lines(frame, hough)
    lines_visualize = display_lines(frame, lines)
    output = cv2.addWeighted(frame, 0.9, lines_visualize, 1, 1)

    histogram = np.sum(frame_new, axis=0)
    midpoint = np.int(histogram.shape[0] / 2)
    # Compute the left and right max pixels
    leftx_ = np.argmax(histogram[:midpoint])
    rightx_ = np.argmax(histogram[midpoint:]) + midpoint
    image_center = int(output.shape[1] / 2)
    # Use the lane pixels to predict the turn
    prediction = turn_predict(image_center, rightx_, leftx_)

    fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)

    cv2.putText(output, 'FPS: ' + str(int(fps)), (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(output, 'Prediction: ' + str(prediction), (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow('frame', output)
    out.write(output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
