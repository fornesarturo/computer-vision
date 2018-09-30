import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10))

while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    histr = cv2.calcHist([gray],[0],None,[256],[0,256])
    ret_th,th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    ax1.clear()
    ax1.imshow(gray,'gray')
    ax1.set_title("Grayscale Image")
    ax1.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    
    ax2.clear()
    ax2.plot(histr)
    ax2.set_title("Histogram")
    ax2.tick_params(top=False, bottom=True, left=False, right=False)

    ax3.clear()
    ax3.imshow(th,'gray')
    ax3.set_title("Binarization (Otsu)")
    ax3.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    plt.pause(0.0001)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()