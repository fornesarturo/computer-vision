import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

fig, axes = plt.subplots(1, 3, figsize=(10, 10)) # y, x

while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
    ret_th, binarized = cv2.threshold(gaussian,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    masked = binarized * gray

    axes[0].clear()
    axes[0].imshow(gray,'gray')
    axes[0].set_title("Grayscale Image")
    axes[0].tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    axes[1].clear()
    axes[1].imshow(binarized,'gray')
    axes[1].set_title("Binarization (Otsu)")
    axes[1].tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    
    axes[2].clear()
    axes[2].imshow(masked,'gray')
    axes[2].set_title("Stitched")
    axes[2].tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    plt.pause(0.0001)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
