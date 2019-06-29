"""

The following code detects a particular colored object and
gives the location of that object.

Also returns the difference between that object and the centre of the camera frame

This particular code is inspired from pyimagesearch.com

"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

#change the range of the desired color (given green)
greenLower = (29, 86, 6) #Hue(0-180) Saturation(0-255) Value(0-255)
greenUpper = (64, 255, 255)
# yeLower = (20, 100, 100)
# yeUpper = (30, 255, 255)

while True:

    _ , frame = cap.read()

    frame = cv2.resize(frame, (640,480))

    blur = cv2.GaussianBlur(frame,(9,9),0)

    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv,greenLower,greenUpper)

    #Opening
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # cnts has two output parameters

    cnts  = cnts[0] #only the first one is needed

    #center of the camera frame
    x1 = int(np.size(frame,0)/2) #300
    y1 = int(np.size(frame,1)/2)

    if len(cnts) > 0:

        c = max(cnts, key=cv2.contourArea) # retrieving the maximum contour area

        ((x, y), radius) = cv2.minEnclosingCircle(c) # location of the detected object

        M = cv2.moments(c)

        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) # center of the object

        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 0, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 0), -1)

        center_diff = (y1-center[0],x1-center[1])

        cv2.putText(frame,'Center:'+str(center),(20,50),cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), lineType=cv2.LINE_AA)

        cv2.putText(frame,'Diff:'+str(center_diff),(20,90),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,200),lineType=cv2.LINE_AA)

    cv2.circle(frame,(y1,x1),3,(233,1,22),-1)

    cv2.imshow('cam',frame)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # print(fps)
    if cv2.waitKey(1) & 0xFF == ord('z'): #ord(char) returns ASCII
        break

cap.release()
cv2.destroyAllWindows()
