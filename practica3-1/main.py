import numpy as np
import imutils
import cv2
import time
import json

from matplotlib import pyplot as plt

def main(debug=False):
    def get_windows(img, increment, window):
        (rows, cols, _) = img.shape
        window_x = window[0]
        window_y = window[1]
        for y in range(0, rows, increment):
            for x in range(0, cols, increment):
                yield (x, y, img[y:y + window_y, x:x + window_x])
    
    def pyramid(image, scale=1.5, minSize=(30, 30)):
        # yield image # don't start with original image as it takes too much time
        while True:
            w = int(image.shape[1] / scale)
            image = imutils.resize(image, width=w)
            if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
                break 
            yield image

    orb = cv2.ORB_create()
    modelImg = cv2.imread("drive_logo.jpg")

    (winW, winH) = (128, 128)

    data = {}

    for img in pyramid(modelImg, scale=2.0):
        # Create new entry for this level of the pyramid
        pyramid_width = img.shape[0]
        data[pyramid_width] = {}

        for (x, y, window) in get_windows(img, 30, (winW, winH)):
            
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
    
            window_kp, window_desc = orb.detectAndCompute(window, None)
            if window_desc is not None:
                print(window_desc.ravel())
                window_key = str(x) + "," + str(y)
                window_desc = window_desc.ravel()
                data[pyramid_width][window_key] = window_desc.tolist()
    
            # draw the window
            clone = img.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv2.imshow("Window", clone)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if debug:
                time.sleep(0.000001)

    cv2.destroyAllWindows()

    with open('data.json', 'w') as outfile:
        json.dump(data, outfile)

if __name__=='__main__':
    main()