import numpy as np
import cv2
from mss import mss
from PIL import Image

bounding_box = {'top': 280, 'left': 500, 'width': 1000, 'height': 1000}
object_dectetor = cv2.createBackgroundSubtractorMOG2(history=200,varThreshold=5,detectShadows=100)
sct = mss()

while True:
    sct_img = sct.grab(bounding_box)
    frame = np.array(sct_img)
    mask = object_dectetor.apply(frame)
    _,mask = cv2.threshold(mask,254, 255, cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    amax = 0
    ax,ay,aw,ah = 280,500,1000,1000
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 2500:
            continue
        cv2.drawContours(frame, [cnt] , -1, (255,0,0),2)
        x,y,w,h = cv2.boundingRect(cnt)
        if area > amax:
            amax = area
            ax,ay,aw,ah = cv2.boundingRect(cnt)
        cv2.rectangle(frame,(x,y), (x+w, y+h),(0, 255, 0),3)
    ax,ay,aw,ah = ax+250,ay+250,500,500
    roi = sct.grab({'top': ay, 'left': ax, 'width': aw, 'height': ah})
    froi = np.array(roi)
    cv2.imshow('roi',froi)
    cv2.imshow('mask',mask)
    cv2.imshow('main',frame)
    
    

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break

