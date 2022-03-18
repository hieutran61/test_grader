from imutils.convenience import grab_contours
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="duong dan toi anh")
args = vars(ap.parse_args())


#xu ly anh
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
edge = cv2.Canny(blur, 100, 200)

#find countours
cnts = cv2.findContours(edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#Ham tinh dien tich hcn bao quanh 1 contour
def S_contours(cnts):
    x,y,w,h = cv2.boundingRect(cnts)
    return w*h

#Ham lay toa do x
def get_x(s):       # s : list co dang [image, (x,y,w,h)]
 return s[1][0]



ans_block = []
x_old, y_old, w_old, h_old = 0, 0, 0, 0

if len(cnts)>0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for i,c in enumerate(cnts):
        x,y,w,h = cv2.boundingRect(c)
        if (w*h) > 100000:
            ans_block.append((gray[y:y+h, x:x+w], [x,y,w,h]))

    
    
    cv2.imshow('Test', ans_block[1][0])











cv2.waitKey()
cv2.destroyAllWindows()



