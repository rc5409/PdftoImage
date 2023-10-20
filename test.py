import numpy as np
import sys
import cv2 as cv
def main(argv):
    
    default_file = 'img/202007211642103000DW_p0-6.png'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(default_file, cv.IMREAD_COLOR)
#     # Check if image is loaded fine
    if src is None:
        print ('Error opening image: ' + argv[0])
        return -1
    # Show source image
    cv.imshow("src", src)
    # [load_image]
    # [gray]
    # Transform source image to gray if it is not already
    if len(src.shape) != 2:
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    else:
        gray = src
    # Show gray image
    show_wait_destroy("gray", gray)
    gray = cv.bitwise_not(gray)
    bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                cv.THRESH_BINARY, 15, -2)
    # Show binary image
    show_wait_destroy("binary", bw)
def show_wait_destroy(winname, img):
    cv.imshow(winname, img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)
if __name__ == "__main__":
    main(sys.argv[1:])