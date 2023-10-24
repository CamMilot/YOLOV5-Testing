import cv2
import torch
from PIL import Image
from mss import mss
import numpy as np
import torch

model = torch.hub.load('ultralytics/yolov5','yolov5s')

sct = mss()

while 1:
    w, h = 1920, 1080
    monitor = {'top': 0, 'left': 0, 'width': w, 'height': h}
    img = Image.frombytes('RGB', (w, h), sct.grab(monitor).rgb)
    screen = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # set the model use the screen
    result = model(screen, size=640)
    print(result)
    cv2.imshow('Screen', result.render()[0])

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break