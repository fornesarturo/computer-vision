import numpy as np
import cv2
import time
from matplotlib import pyplot as plt

no_of_frames = 10
frames = 0

ball = [ 0, 1920//2 ]
radius = 4

# fig, axes = plt.subplots(1, 3, figsize=(15, 15))  # y, x
cap = cv2.VideoCapture('20181001_202407.mp4')
kernel = np.ones((5, 5), np.uint8)
red = np.array([0, 0, 255])
blue = np.array([255, 0, 0])

while(cap.isOpened()):
    frames += 1
    ret, frame = cap.read()

    # print(frame.shape)

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret_th, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    masked = binarized * gray
    negation = cv2.bitwise_not(binarized)
    dilation = cv2.dilate(negation, kernel, iterations=2)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

    collides = False

    for i in range(ball[0] - 20, ball[0] + 20):
        for j in range(ball[1] - 20, ball[1] + 20):
            # is inside the size
            if i >= 0 and j >= 0 and i < 1080 and j < 1920:
                if closing[i][j] == 255:
                    collides = True
                    break
        if collides:
            break
    
    if collides:
        color = blue
    else:
        color = red

    for i in range(ball[0] - 20, ball[0] + 20):
        for j in range(ball[1] - 20, ball[1] + 20):
            # is inside the size
            if i >= 0 and j >= 0 and i < 1080 and j < 1920:
                frame[i][j] = color


    ball[0] += 10
    # ball[1] += 10

    cv2.imshow('binarized', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
