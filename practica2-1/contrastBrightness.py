import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

fig, axes = plt.subplots(2, 2, figsize=(10, 10)) # subplots y, x

while(True):
    ret, frame = cap.read()
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_th,th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    contrast = 1.5
    brightness = 30

    # new_img = frame * alpha + beta
    mul_img = cv2.multiply(color, np.array([contrast]))
    new_img = cv2.add(mul_img, brightness)

    axes[0][0].clear()
    axes[0][0].imshow(gray,'gray')
    axes[0][0].set_title("Grayscale Frame")
    axes[0][0].tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    axes[0][1].clear()
    axes[0][1].imshow(th,'gray')
    axes[0][1].set_title("Binarization (Otsu)")
    axes[0][1].tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    axes[1][0].clear()
    axes[1][0].imshow(color)
    axes[1][0].set_title("Original Frame")
    axes[1][0].tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    
    axes[1][1].clear()
    axes[1][1].imshow(new_img)
    axes[1][1].set_title("Contrast + Brightness modified")
    axes[1][1].tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    plt.pause(0.0001)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
