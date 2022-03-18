from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="duong dan toi anh")
args = vars(ap.parse_args())


ANSWER_KEY = {0:1, 1:4, 2:0, 3:3, 4:1}
RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(blur, 75, 200)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None
if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

docCnt = None
for c in cnts:
    #Tinh chu vi cua contours va tim duong vien xap xi
    peri = cv2.arcLength(c, closed=True)
    approx = cv2.approxPolyDP(c, 0.02*peri, closed=True)

    #Neu duong vien xap xi la hinh chu nhat (4 canh) thi xac dinh dc do la to giay
    if len(approx) == 4:     
        docCnt = approx
        break

#lay ra hinh anh bai thi tu anh goc
paper = four_point_transform(image, docCnt.reshape(4,2))
warped = four_point_transform(gray, docCnt.reshape(4,2))

#nhi phan buc anh
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []


for c in cnts:
    (x,y,w,h) = cv2.boundingRect(c)
    aspect = w/ float(h)
    if w>=20 and h>= 20 and aspect>=0.9 and aspect<=1.1:
        questionCnts.append(c)

questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
correct = 0




for (q,i) in enumerate(np.arange(0, len(questionCnts), 5)):
    cnts = contours.sort_contours(questionCnts[i:i+5], method="left-to-right")[0]
    bubbled = None
    count = 0
    
    for (j,c) in enumerate(cnts):
        
        
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)
        print(total)
        
        if total > 600 :
            bubbled = j
            count+=1
    
    color = RED
    k = ANSWER_KEY[q]
    if k == bubbled and count == 1:
        color = GREEN
        correct +=1
    cv2.drawContours(paper, [cnts[k]], -1, color, 2)
    
    
   
    

score = (correct/len(ANSWER_KEY))  * 10

        
cv2.putText(paper, f'{score:.2f}', (280,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2)        
cv2.imshow('Paper', paper)

        
cv2.imshow('Thresh', thresh)    
    
    







cv2.waitKey()
cv2.destroyAllWindows()