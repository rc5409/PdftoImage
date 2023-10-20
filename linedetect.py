import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = "img/202007211642103000DW_p0-6.png"
image = cv2.imread(image_path)
cv2.imshow("graph", image)


def grey(image):
    image = np.asarray(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def gauss(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def canny(image):
    edges = cv2.Canny(image, 50, 150)
    return edges


def region(image):
    height, width = image.shape
    # isolate the gradients that correspond to the lane lines
    triangle = np.array([
        [(100, height), (475, 325), (width, height)]
    ])
    # create a black image with the same dimensions as original image
    mask = np.zeros_like(image)
    # create a mask (triangle that isolates the region of interest in our image)
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask


def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    # make sure array isn't empty
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            # draw lines on a black image
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
            cv2.imshow('line', lines_image)
    return lines_image


def make_points(image, line_parameters):
    slope, intercept = line_parameters
    y1 = int(image.shape[0])
    y2 = int(y1*3/5)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    return np.array((left_line, right_line))


# copy = np.copy(image1)
copy = np.copy(image)
edges = cv2.Canny(copy, 50, 150)
isolated = region(edges)
cv2.imshow("edge", edges)
cv2.imshow("isolated", isolated)
cv2.waitKey(0)


# DRAWING LINES: (order of params) --> region of interest, bin size (P, theta), min intersections needed, placeholder array,
lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100,
                        np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = average_slope_intercept(copy, lines)
black_lines = display_lines(copy, averaged_lines)
# taking wighted sum of original image and lane lines image
final = cv2.addWeighted(copy, 0.8, black_lines, 1, 1)
cv2.imshow("final", final)
cv2.waitKey(0)
