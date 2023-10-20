import cv2, imutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pathlib import Path
from matplotlib import rcParams

img_dir = 'img'
save_dir = 'out'
def is_contour_bad(c):
    # approximate the contour.
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # the contour is 'bad' if it is not a 4-sided.
    return not len(approx) != 4
for path in Path(img_dir).iterdir():
    if path.name.endswith('.png') or path.name.endswith('.jpg') or path.name.endswith('.jpeg'):
        filepath = img_dir + "/" + path.name
        image = cv2.imread(filepath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
        
        # find external contours in the thresholded image and..
        # allocate memory for the convex hull image.
        
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        hullImage = np.zeros(gray.shape[:2], dtype="uint8")
        
        areaList = []
        xes = []
        # loop over the contours.
        for (i, c) in enumerate(cnts):
            #if the contours are bad, ignore those contours.
            if is_contour_bad(c):
                # compute the area of the contour along with the bounding box.
                area = cv2.contourArea(c)
                (x, y, w, h) = cv2.boundingRect(c)

                # compute the convex hull of the contour..
                # then compute the area of the convex hull.

                hull = cv2.convexHull(c)

                hullArea = cv2.contourArea(hull)
#               solidity = area / float(hullArea)

               # draw contours.
                cv2.drawContours(hullImage, [hull], -1, 255, -1)
                cv2.drawContours(image, [c], -1, (240, 0, 159), 3)
                areaList.append([x,y,w,h])
                xes.append(x)
#         print(sorted(areaList))
        rcParams['figure.figsize'] = 15, 8

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(hullImage, aspect = 'auto')
        ax[1].imshow(image, aspect = 'auto')

        cv2.imshow('hull',hullImage)
        cv2.waitKey(0)
        
        #cv2.imshow('output',image)
def findMaxConsecutiveOnes(nums) -> int:
    count = maxCount = 0
    
    for i in range(len(nums)):
        if nums[i] == 1:
            count += 1
        else:
            maxCount = max(count, maxCount)
            count = 0
                
    return max(count, maxCount)
def detectAxes(filepath, threshold=None, debug=False):
    if filepath is None:
        return None, None
    
    if threshold is None:
        threshold = 10
    
    image = cv2.imread(filepath)
    height, width, channels = image.shape
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get the max-consecutive-ones for eah column in the bw image, and...
    # pick the "first" index that fall in [max - threshold, max + threshold]
    maxConsecutiveOnes = [findMaxConsecutiveOnes(gray[:, idx] < 200) for idx in range(width)]
    start_idx, maxindex, maxcount = 0, 0, max(maxConsecutiveOnes)
    while start_idx < width:
        if abs(maxConsecutiveOnes[start_idx] - maxcount) <= threshold:
            maxindex = start_idx
            break
            
        start_idx += 1
           
    yaxis = (maxindex, 0, maxindex, height)
    
    if debug:
        fig, ax = plt.subplots(1, 2)

        ax[0].imshow(image)

        ax[1].plot(maxConsecutiveOnes, color = 'k')
        ax[1].axhline(y = max(maxConsecutiveOnes) - 10, color = 'r', linestyle = 'dashed')
        ax[1].axhline(y = max(maxConsecutiveOnes) + 10, color = 'r', linestyle = 'dashed')
        ax[1].vlines(x = maxindex, ymin = 0.0, ymax = maxConsecutiveOnes[maxindex], color = 'b', linewidth = 4)

        plt.show()

    # Get the max-consecutive-ones for eah row in the bw image, and...
    # pick the "last" index that fall in [max - threshold, max + threshold]
    maxConsecutiveOnes = [findMaxConsecutiveOnes(gray[idx, :] < 200) for idx in range(height)]
    start_idx, maxindex, maxcount = 0, 0, max(maxConsecutiveOnes)
    while start_idx < height:
        if abs(maxConsecutiveOnes[start_idx] - maxcount) <= threshold:
            maxindex = start_idx
            
        start_idx += 1
            
    cv2.line(image, (0, maxindex), (width, maxindex),  (255, 0, 0), 2)
    xaxis = (0, maxindex, width, maxindex)
    
    if debug:
        rcParams['figure.figsize'] = 15, 8

        fig, ax = plt.subplots(1, 1)
        ax.imshow(image, aspect = 'auto')
        
    return xaxis, yaxis
for path in Path(img_dir).iterdir():
    filepath = img_dir + "/" + path.name
    image = cv2.imread(filepath)
    xaxis, yaxis = detectAxes(filepath)

    for (x1, y1, x2, y2) in [xaxis]:
        cv2.line(image, (x1, y1), (x2, y2),  (0, 0, 255), 2)

    for (x1, y1, x2, y2) in [yaxis]:
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imwrite(save_dir + '/' + ''.join(path.name.split(".")[:-1]) + "_axes.png", image)
# badplots = []

# for path in Path("out").iterdir():
#     badplots.append(path.name)

# f, ax = plt.subplots(4, 4, figsize = (20, 20))

# for index in range(16):
#     ax[index // 4, index % 4].imshow(mpimg.imread("./out/bad/" + badplots[index]))
#     ax[index // 4, index % 4].axis('off')
#     ax[index // 4, index % 4].set_aspect('equal')
    
# plt.show()