# Adapted from code from Mutaza's Workshop

import cv2
import numpy as np
import os
import HandTrackingModule as htm
import random
from datetime import date
from time import time

folderPath = "Header"
myList = os.listdir(folderPath)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

# Set default values
header = overlayList[0]
headerSelected = 1
drawColour = (0,0,255)
brushThickness = 10

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0,0
imageCanvas = np.zeros((720,1280,3),np.uint8)

screenshotTaken = False

while True:

    # 1. Import the image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
       
        x1, y1 = lmList[8][1:] # Index fingertip
        x2, y2 = lmList[12][1:] # Middle fingertip
        x3, y3 = lmList[16][1:] # Ring fingertip
        x4, y4 = lmList[20][1:] # Pinky fingertip

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        # 4. If Selection mode 
        if fingers[1] and fingers[2] and fingers[3] and fingers[4]:
            screenshotTaken = False
            xp, yp = 0,0
            cv2.rectangle(img, (x1,y1-25), (x2,y2+25), drawColour, cv2.FILLED)

            # Checking for colour change
            if y1 < 125:
                if 110 < x1 < 175:
                    header = overlayList[0]
                    headerSelected = 1
                    drawColour = (0,0,255)
                elif 250 < x1 < 325:
                    header = overlayList[1]
                    headerSelected = 2
                    drawColour = (0,255,255)
                elif 400 < x1 < 465:
                    header = overlayList[2]
                    headerSelected = 3
                    drawColour = (0,255,0)
                elif 540 < x1 < 605:
                    header = overlayList[3]
                    headerSelected = 4
                    drawColour = (255,0,0)
                elif 680 < x1 < 745:
                    header = overlayList[4]
                    headerSelected = 5
                    drawColour = (0,165,255)
                elif 820 < x1 < 885:
                    header = overlayList[5]
                    headerSelected = 6
                elif 960 < x1 < 1025:
                    header = overlayList[6]
                    headerSelected = 7
                elif 1100 < x1 < 1165:
                    header = overlayList[7]
                    headerSelected = 8
                    drawColour = (0,0,0)


        
        # 4. If Picture mode
        if fingers[1] == False and fingers[2] == False and fingers[3] == False and fingers[4] == False:
            if(screenshotTaken == False):
                # Creates mask by making images black and white, then merging them
                imageGrey = cv2.cvtColor(imageCanvas, cv2.COLOR_BGR2GRAY)
                _, imgInv = cv2.threshold(imageGrey, 50, 255, cv2.THRESH_BINARY_INV)
                imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
                img = cv2.bitwise_and(img, imgInv)
                img = cv2.bitwise_or(img, imageCanvas)
                screenshotName = "Screenshot {} {}.png".format(date.today(), int(time() * 1000))
                cv2.imwrite(str(screenshotName), img)
                print("Screenshot taken")
                screenshotTaken = True


        # 5. If Drawing mode
        if fingers[4] == False and fingers[1]:
            screenshotTaken = False
            otherColour = random.randrange(0,255)

            # Sets drawing parameters
            if fingers[1] and fingers[2] == False and fingers[3] == False:
                brushThickness = 10
                fingerTip = (x1,y1)
            elif fingers[1] and fingers[2] and fingers[3] == False:
                brushThickness = 15
                fingerTip = (x2, y2)
            else:
                brushThickness = 20
                fingerTip = (x2, y2)
            
            # Makes sure line isn't drawn from the origin
            if xp == 0 and yp == 0:
                xp, yp = fingerTip
            
            # Handles whether motion colours should be drawn or not
            # Outside of selection mode, otherwise it will be a single randomly generated colour
            if headerSelected == 6:
                drawColour = (0,otherColour,255)
            elif headerSelected == 7:
                drawColour = (255,otherColour,otherColour)
            
            # Actually draws the colour
            cv2.circle(img, fingerTip,brushThickness,drawColour,cv2.FILLED)
            cv2.line(img, (xp, yp), fingerTip, drawColour,brushThickness)
            cv2.line(imageCanvas, (xp, yp), fingerTip, drawColour,brushThickness)
                
            # Makes current tip position to previous, so more lines can be drawn
            xp, yp = fingerTip

    # Creates mask by making images black and white, then merging them
    imageGrey = cv2.cvtColor(imageCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imageGrey, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imageCanvas)

    # Setting the header image
    img[0:125, 0: 1280] = header
    cv2.imshow("Image",img)
    cv2.waitKey(1)