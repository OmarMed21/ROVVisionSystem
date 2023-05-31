import numpy as np
import cv2 as cv
img = cv.imread('Iphone12.jpg')
img = cv.resize(img, (int(img.shape[1]//2), int(img.shape[0]//2)),interpolation=cv.INTER_AREA)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#cv.imshow('OMAR', gray)
ret, thresh = cv.threshold(gray, 127, 255, 0)
blur = cv.GaussianBlur(gray, (7,7), 1)
canny = cv.Canny(blur, 60, 60)
kernel = np.ones((5,5))
dial = cv.dilate(canny, kernel, iterations=3)
erode = cv.erode(dial, kernel ,iterations=2)
contours, hierarchy = cv.findContours(erode, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

final = []
for i in contours:
    area = cv.contourArea(i)
    if area > 1000:
        perci = cv.arcLength(i, True)
        approx = cv.approxPolyDP(i, .02*perci, True)
        bbx = cv.boundingRect(approx)
        if len(approx) == 0:
            final.append(len(approx), area, approx, bbx, i)

final = sorted(final, key = lambda x:x[1], reverse=True)

for con in final:
    cv.drawContours(img, con[4], -1, (0,0,255),3)

cv.imshow('c',final)
cv.waitKey(0)
