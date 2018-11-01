import numpy as np
import cv2

cap = cv2.VideoCapture("puntos.mp4")

green_min = 70
green_max = 90

kernel = np.ones((9, 9), np.uint8)

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

polyline_color = (0, 255, 255)

while(True):
    ret, frame = cap.read()

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

    im_with_keypoints = cv2.drawKeypoints(green_mask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    hull = cv2.convexHull(pts, False)
    # cv2.drawContours(frame, [hull], 0, (0, 255, 0), 10, 8)
    
    board_mask = np.zeros(frame.shape, np.uint8)
    cv2.fillPoly(board_mask, [hull], (255, 255, 255))

    green = frame & board_mask

    # rgb_masked = cv2.bitwise_and(rgb, rgb, mask=closing)

    cv2.imshow('binarized', green)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
