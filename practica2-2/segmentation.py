import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

fig, axes = plt.subplots(2, 3, figsize=(11, 8))  # subplots y, x

red_min = np.array([160, 20, 70])
red_max = np.array([190, 255, 255])

blue_min = np.array([95, 0, 50])
blue_max = np.array([122, 255, 255])

green_min = np.array([70, 0, 0])
green_max = np.array([100, 255, 255])

kernel = np.ones((9, 9), np.uint8)

while(True):
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    red_mask = cv2.inRange(hsv, red_min, red_max)
    green_mask = cv2.inRange(hsv, green_min, green_max)
    blue_mask = cv2.inRange(hsv, blue_min, blue_max)

    red = cv2.bitwise_and(rgb, rgb, mask=red_mask)
    green = cv2.bitwise_and(rgb, rgb, mask=green_mask)
    blue = cv2.bitwise_and(rgb, rgb, mask=blue_mask)

    rg_mask = cv2.bitwise_or(red_mask, green_mask)
    rgb_mask = cv2.bitwise_or(rg_mask, blue_mask)

    opening = cv2.morphologyEx(rgb_mask, cv2.MORPH_OPEN, kernel)

    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    rgb_masked = cv2.bitwise_and(rgb, rgb, mask=closing)

    axes[0][0].clear()
    axes[0][0].imshow(rgb)
    axes[0][0].set_title("Original")
    axes[0][0].tick_params(labelcolor='w', top=False,
                           bottom=False, left=False, right=False)

    axes[0][2].clear()
    axes[0][2].imshow(rgb_masked)
    axes[0][2].set_title("Three objects")
    axes[0][2].tick_params(labelcolor='w', top=False,
                           bottom=False, left=False, right=False)
    
    axes[1][0].clear()
    axes[1][0].imshow(red)
    axes[1][0].set_title("Red")
    axes[1][0].tick_params(labelcolor='w', top=False,
                           bottom=False, left=False, right=False)
    
    axes[1][1].clear()
    axes[1][1].imshow(green)
    axes[1][1].set_title("Green")
    axes[1][1].tick_params(labelcolor='w', top=False,
                           bottom=False, left=False, right=False)
    
    axes[1][2].clear()
    axes[1][2].imshow(blue)
    axes[1][2].set_title("Blue")
    axes[1][2].tick_params(labelcolor='w', top=False,
                           bottom=False, left=False, right=False)
    

    plt.pause(0.0001)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
