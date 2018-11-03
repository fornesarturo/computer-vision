import numpy as np
import cv2

def getROI(frame, kernel, detector, debug=False):
    green_min = 70
    green_max = 90

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    h, s, v = cv2.split(hsv)

    green_mask = cv2.inRange(h, green_min, green_max)

    green_mask = cv2.erode(green_mask, kernel, iterations=2)
    green_mask = cv2.dilate(green_mask, kernel, iterations=1)

    keypoints = detector.detect(green_mask)

    points = []
    for keypoint in keypoints:
        points.append([keypoint.pt[0], keypoint.pt[1]])

    if debug:
        im_with_keypoints = cv2.drawKeypoints(green_mask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    hull = cv2.convexHull(pts, False)
    
    if debug:
        cv2.drawContours(im_with_keypoints, [hull], 0, (0, 255, 0), 10, 8)
    
    board_mask = np.zeros(frame.shape, np.uint8)
    cv2.fillPoly(board_mask, [hull], (255, 255, 255))

    roi = frame & board_mask

    if debug:
        return roi, im_with_keypoints
    return roi, False

def binarize(frame, kernel):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret_th, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    masked = binarized * gray
    negation = cv2.bitwise_not(binarized)
    dilation = cv2.dilate(negation, kernel, iterations=2)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

    return closing

def getBlobDetector():
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    params.filterByCircularity = True
    params.minCircularity = 0.6

    params.filterByArea = True
    params.minArea = 200

    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByColor = False

    detector = cv2.SimpleBlobDetector_create(params)
    
    return detector

def main(cap, debug=False):
    detector = getBlobDetector()

    kernel_blobs = np.ones((9, 9), np.uint8)
    kernel_binarization = np.ones((5, 5), np.uint8)

    while(cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            break

        roi, debug_im = getROI(frame, kernel_blobs, detector)
        binarized = binarize(roi, kernel_binarization)

        cv2.imshow('ROI', binarized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    cap = cv2.VideoCapture("puntos.mp4")
    main(cap)
