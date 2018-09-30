import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

fig, axes = plt.subplots(2, 3) # subplots y, x

while(True):
    ret, frame = cap.read()

    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0)

    # Canny
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gaussian)
    canny = cv2.Canny(gaussian, minVal, maxVal)

    # Sobel
    # Gradient X
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    # Gradient Y
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    # Prewitt
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    img_prewittx = cv2.filter2D(gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(gaussian, -1, kernely)
    prewitt = img_prewittx + img_prewitty

    axes[0][0].clear()
    axes[0][0].imshow(color)
    axes[0][0].set_title("Original")
    axes[0][0].tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    axes[0][1].clear()
    axes[0][1].imshow(gray, 'gray')
    axes[0][1].set_title("Grayscale")
    axes[0][1].tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    
    axes[0][2].clear()
    axes[0][2].imshow(gaussian, 'gray')
    axes[0][2].set_title("Grayscale (Gauss reduced noise)")
    axes[0][2].tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    axes[1][0].clear()
    axes[1][0].imshow(sobel, 'gray')
    axes[1][0].set_title("Sobel")
    axes[1][0].tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    axes[1][1].clear()
    axes[1][1].imshow(prewitt, 'gray')
    axes[1][1].set_title("Prewitt")
    axes[1][1].tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    axes[1][2].clear()
    axes[1][2].imshow(canny, 'gray')
    axes[1][2].set_title("Canny")
    axes[1][2].tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    plt.pause(0.0001)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
